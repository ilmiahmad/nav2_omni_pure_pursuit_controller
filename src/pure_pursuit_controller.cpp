// Copyright (c) 2020 Shrijit Singh
// Copyright (c) 2020 Samsung Research America
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_pure_pursuit_controller/pure_pursuit_controller.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nav2_core/exceptions.hpp"
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_util/node_utils.hpp"

using nav2_util::declare_parameter_if_not_declared;
using nav2_util::geometry_utils::euclidean_distance;
using std::abs;
using std::hypot;
using std::max;
using std::min;
using namespace nav2_costmap_2d;  // NOLINT
using rcl_interfaces::msg::ParameterType;

namespace nav2_pure_pursuit_controller {

void PurePursuitController::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent, std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) {
  auto node = parent.lock();
  node_ = parent;
  if (!node) {
    throw nav2_core::PlannerException("Unable to lock node!");
  }

  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  tf_ = tf;
  plugin_name_ = name;
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  double transform_tolerance = 1.0;
  double control_frequency = 20.0;
  use_interpolation_ = false;
  lookahead_distance_ = 0.3;
  //goal_dist_tol_ = 0.25;  // reasonable default before first update

  declare_parameter_if_not_declared(node, plugin_name_ + ".moving_kp",
                                    rclcpp::ParameterValue(3.0));
  declare_parameter_if_not_declared(node, plugin_name_ + ".moving_ki",
                                    rclcpp::ParameterValue(0.1));
  declare_parameter_if_not_declared(node, plugin_name_ + ".moving_kd",
                                    rclcpp::ParameterValue(0.3));
  declare_parameter_if_not_declared(node, plugin_name_ + ".heading_kp",
                                    rclcpp::ParameterValue(3.0));
  declare_parameter_if_not_declared(node, plugin_name_ + ".heading_ki",
                                    rclcpp::ParameterValue(0.1));
  declare_parameter_if_not_declared(node, plugin_name_ + ".heading_kd",
                                    rclcpp::ParameterValue(0.3));
  declare_parameter_if_not_declared(node, plugin_name_ + ".transform_tolerance",
                                    rclcpp::ParameterValue(0.1));
  declare_parameter_if_not_declared(
      node, plugin_name_ + ".min_max_sum_error",
      rclcpp::ParameterValue(1.0));
  declare_parameter_if_not_declared(
      node, plugin_name_ + ".max_robot_pose_search_dist",
      rclcpp::ParameterValue(0.5));
      
  node->get_parameter(plugin_name_ + ".moving_kp",
                      moving_kp_);
  node->get_parameter(plugin_name_ + ".moving_ki", moving_ki_);
  node->get_parameter(plugin_name_ + ".moving_kd",
                      moving_kd_);
  node->get_parameter(plugin_name_ + ".heading_kp",
                      heading_kp_);
  node->get_parameter(plugin_name_ + ".heading_ki", heading_ki_);
  node->get_parameter(plugin_name_ + ".heading_kd",
                      heading_kd_);
  node->get_parameter(plugin_name_ + ".transform_tolerance",
                      transform_tolerance);
  node->get_parameter(plugin_name_ + ".min_max_sum_error",
                      min_max_sum_error_);
  node->get_parameter(plugin_name_ + ".lookahead_distance",
                      lookahead_distance_);

  node->get_parameter("controller_frequency", control_frequency);

  transform_tolerance_ = tf2::durationFromSec(transform_tolerance);
  control_duration_ = 1.0 / control_frequency;

  global_path_pub_ =
      node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);
  carrot_pub_ = node->create_publisher<geometry_msgs::msg::PointStamped>(
      "lookahead_point", 1);
  carrot_arc_pub_ =
      node->create_publisher<nav_msgs::msg::Path>("lookahead_collision_arc", 1);

  // initialize collision checker and set costmap
  collision_checker_ = std::make_unique<
      nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>>(
      costmap_);
  collision_checker_->setCostmap(costmap_);

  move_pid = std::make_shared<PID>(control_duration_, 2, -2, moving_kp_, moving_kd_, moving_ki_);
  heading_pid = std::make_shared<PID>(control_duration_, 2, -2, heading_kp_, heading_kd_, heading_ki_);
}

void PurePursuitController::cleanup() {
  RCLCPP_INFO(logger_,
              "Cleaning up controller: %s of type"
              " regulated_pure_pursuit_controller::PurePursuitController",
              plugin_name_.c_str());
  global_path_pub_.reset();
  carrot_pub_.reset();
  carrot_arc_pub_.reset();
}

void PurePursuitController::activate() {
  RCLCPP_INFO(logger_,
              "Activating controller: %s of type "
              "regulated_pure_pursuit_controller::PurePursuitController",
              plugin_name_.c_str());
  global_path_pub_->on_activate();
  carrot_pub_->on_activate();
  carrot_arc_pub_->on_activate();
  // Add callback for dynamic parameters
  auto node = node_.lock();
  dyn_params_handler_ = node->add_on_set_parameters_callback(
      std::bind(&PurePursuitController::dynamicParametersCallback, this,
                std::placeholders::_1));
}

void PurePursuitController::deactivate() {
  RCLCPP_INFO(logger_,
              "Deactivating controller: %s of type "
              "regulated_pure_pursuit_controller::PurePursuitController",
              plugin_name_.c_str());
  global_path_pub_->on_deactivate();
  carrot_pub_->on_deactivate();
  carrot_arc_pub_->on_deactivate();
  dyn_params_handler_.reset();
}


geometry_msgs::msg::TwistStamped PurePursuitController::computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped &pose,
    const geometry_msgs::msg::Twist & /*velocity*/,
    nav2_core::GoalChecker * /*goal_checker*/) {
  std::lock_guard<std::mutex> lock_reinit(mutex_);

  nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap->getMutex()));

  // Transform path to robot base frame
  auto transformed_plan = transformGlobalPlan(pose);

  
  auto carrot_pose = getLookAheadPoint(lookahead_distance_, transformed_plan);
  // Find distance^2 to look ahead point (carrot) in robot base frame
  // This is the chord length of the circle
  double lin_dist = hypot(carrot_pose.pose.position.x, carrot_pose.pose.position.y);
  double theta_dist = atan2(carrot_pose.pose.position.y, carrot_pose.pose.position.x);
  double heading_dist = carrot_pose.pose.orientation.z;

  RCLCPP_INFO(logger_,
              "linear : %lf heading : %lf",
              lin_dist ,heading_dist);

  auto lin_vel = move_pid->calculate(lin_dist,0);
  auto angular_vel = heading_pid->calculate(heading_dist,0);
  // populate and return message
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header = pose.header;
  cmd_vel.twist.linear.x = lin_vel*cos(theta_dist);
  cmd_vel.twist.linear.y = lin_vel*sin(theta_dist);
  cmd_vel.twist.angular.z = angular_vel;
  return cmd_vel;
}

double PurePursuitController::angleWrap(double angle) {
    
    if(angle >= M_PI)
    {
      angle -= 2*M_PI;
    }

    else if(angle < -M_PI)
    {
      angle += 2*M_PI;
    }

    return angle;
}

void PurePursuitController::setPlan(const nav_msgs::msg::Path &path) {
  global_plan_ = path;
}

void PurePursuitController::setSpeedLimit(const double &/*speed_limit*/,
                                          const bool &/*percentage*/) {
  // if (speed_limit == nav2_costmap_2d::NO_SPEED_LIMIT) {
  //   // Restore default value
  //   desired_linear_vel_ = base_desired_linear_vel_;
  // } else {
  //   if (percentage) {
  //     // Speed limit is expressed in % from maximum speed of robot
  //     desired_linear_vel_ = base_desired_linear_vel_ * speed_limit / 100.0;
  //   } else {
  //     // Speed limit is expressed in absolute value
  //     desired_linear_vel_ = speed_limit;
  //   }
  // }
}


nav_msgs::msg::Path PurePursuitController::transformGlobalPlan(
    const geometry_msgs::msg::PoseStamped &pose) {
  if (global_plan_.poses.empty()) {
    throw nav2_core::PlannerException("Received plan with zero length");
  }

  // let's get the pose of the robot in the frame of the plan
  geometry_msgs::msg::PoseStamped robot_pose;
  if (!transformPose(global_plan_.header.frame_id, pose, robot_pose)) {
    throw nav2_core::PlannerException(
        "Unable to transform robot pose into global plan's frame");
  }

  // We'll discard points on the plan that are outside the local costmap
  double max_costmap_extent = getCostmapMaxExtent();
  // double max_robot_pose_search_dist_ = 0.5;
  // auto closest_pose_upper_bound =
  //     nav2_util::geometry_utils::first_after_integrated_distance(
  //         global_plan_.poses.begin(), global_plan_.poses.end(),
  //         max_robot_pose_search_dist_);

  // First find the closest pose on the path to the robot
  // bounded by when the path turns around (if it does) so we don't get a pose
  // from a later portion of the path
  auto transformation_begin = nav2_util::geometry_utils::min_by(
      global_plan_.poses.begin(), global_plan_.poses.end(),
      [&robot_pose](const geometry_msgs::msg::PoseStamped &ps) {
        return euclidean_distance(robot_pose, ps);
      });

  // Find points up to max_transform_dist so we only transform them.
  auto transformation_end = std::find_if(
      transformation_begin, global_plan_.poses.end(), [&](const auto &pose) {
        return euclidean_distance(pose, robot_pose) > max_costmap_extent;
      });

  // Lambda to transform a PoseStamped from global frame to local
  auto transformGlobalPoseToLocal = [&](const auto &global_plan_pose) {
    geometry_msgs::msg::PoseStamped stamped_pose, transformed_pose;
    stamped_pose.header.frame_id = global_plan_.header.frame_id;
    stamped_pose.header.stamp = robot_pose.header.stamp;
    stamped_pose.pose = global_plan_pose.pose;
    transformPose(costmap_ros_->getBaseFrameID(), stamped_pose,
                  transformed_pose);
    transformed_pose.pose.position.z = 0.0;
    return transformed_pose;
  };

  
  // Transform the near part of the global plan into the robot's frame of
  // reference.
  nav_msgs::msg::Path transformed_plan;

  std::transform(transformation_begin, transformation_end,
                 std::back_inserter(transformed_plan.poses),
                 transformGlobalPoseToLocal);

  transformed_plan.header.frame_id = costmap_ros_->getBaseFrameID();
  transformed_plan.header.stamp = robot_pose.header.stamp;

  // Remove the portion of the global plan that we've already passed so we don't
  // process it on the next iteration (this is called path pruning)
  global_plan_.poses.erase(begin(global_plan_.poses), transformation_begin);
  global_path_pub_->publish(transformed_plan);

  if (transformed_plan.poses.empty()) {
    throw nav2_core::PlannerException("Resulting plan has 0 poses in it.");
  }

  return transformed_plan;
  
}

std::unique_ptr<geometry_msgs::msg::PointStamped> PurePursuitController::createCarrotMsg(
  const geometry_msgs::msg::PoseStamped & carrot_pose)
{
  auto carrot_msg = std::make_unique<geometry_msgs::msg::PointStamped>();
  carrot_msg->header = carrot_pose.header;
  carrot_msg->point.x = carrot_pose.pose.position.x;
  carrot_msg->point.y = carrot_pose.pose.position.y;
  carrot_msg->point.z = 0.01;  // publish right over map to stand out
  return carrot_msg;
}

geometry_msgs::msg::PoseStamped PurePursuitController::getLookAheadPoint(
  const double & lookahead_dist,
  const nav_msgs::msg::Path & transformed_plan)
{
  // Find the first pose which is at a distance greater than the lookahead distance
  auto goal_pose_it = std::find_if(
    transformed_plan.poses.begin(), transformed_plan.poses.end(), [&](const auto & ps) {
      return hypot(ps.pose.position.x, ps.pose.position.y) >= lookahead_dist;
    });

  // If the no pose is not far enough, take the last pose
  if (goal_pose_it == transformed_plan.poses.end()) {
    goal_pose_it = std::prev(transformed_plan.poses.end());
    //goal_pose_it = transformed_plan.poses.begin();
  } 
  //   else if (use_interpolation_ && goal_pose_it != transformed_plan.poses.begin()) {
  //   // Find the point on the line segment between the two poses
  //   // that is exactly the lookahead distance away from the robot pose (the origin)
  //   // This can be found with a closed form for the intersection of a segment and a circle
  //   // Because of the way we did the std::find_if, prev_pose is guaranteed to be inside the circle,
  //   // and goal_pose is guaranteed to be outside the circle.
  //   auto prev_pose_it = std::prev(goal_pose_it);
  //   auto point = circleSegmentIntersection(
  //     prev_pose_it->pose.position,
  //     goal_pose_it->pose.position, lookahead_dist);
  //   geometry_msgs::msg::PoseStamped pose;
  //   pose.header.frame_id = prev_pose_it->header.frame_id;
  //   pose.header.stamp = goal_pose_it->header.stamp;
  //   pose.pose.position = point;
  //   return pose;
  // }

  return *goal_pose_it;
}

geometry_msgs::msg::Point PurePursuitController::circleSegmentIntersection(
  const geometry_msgs::msg::Point & p1,
  const geometry_msgs::msg::Point & p2,
  double r)
{
  // Formula for intersection of a line with a circle centered at the origin,
  // modified to always return the point that is on the segment between the two points.
  // https://mathworld.wolfram.com/Circle-LineIntersection.html
  // This works because the poses are transformed into the robot frame.
  // This can be derived from solving the system of equations of a line and a circle
  // which results in something that is just a reformulation of the quadratic formula.
  // Interactive illustration in doc/circle-segment-intersection.ipynb as well as at
  // https://www.desmos.com/calculator/td5cwbuocd
  double x1 = p1.x;
  double x2 = p2.x;
  double y1 = p1.y;
  double y2 = p2.y;

  double dx = x2 - x1;
  double dy = y2 - y1;
  double dr2 = dx * dx + dy * dy;
  double D = x1 * y2 - x2 * y1;

  // Augmentation to only return point within segment
  double d1 = x1 * x1 + y1 * y1;
  double d2 = x2 * x2 + y2 * y2;
  double dd = d2 - d1;

  geometry_msgs::msg::Point p;
  double sqrt_term = std::sqrt(r * r * dr2 - D * D);
  p.x = (D * dy + std::copysign(1.0, dd) * dx * sqrt_term) / dr2;
  p.y = (-D * dx + std::copysign(1.0, dd) * dy * sqrt_term) / dr2;
  return p;
}


double PurePursuitController::getCostmapMaxExtent() const
{
  const double max_costmap_dim_meters = std::max(
    costmap_->getSizeInMetersX(), costmap_->getSizeInMetersY());
  return max_costmap_dim_meters / 2.0;
}
bool PurePursuitController::transformPose(
    const std::string frame, const geometry_msgs::msg::PoseStamped &in_pose,
    geometry_msgs::msg::PoseStamped &out_pose) const {
  if (in_pose.header.frame_id == frame) {
    out_pose = in_pose;
    return true;
  }

  try {
    tf_->transform(in_pose, out_pose, frame, transform_tolerance_);
    out_pose.header.frame_id = frame;
    return true;
  } catch (tf2::TransformException &ex) {
    RCLCPP_ERROR(logger_, "Exception in transformPose: %s", ex.what());
  }
  return false;
}

rcl_interfaces::msg::SetParametersResult
PurePursuitController::dynamicParametersCallback(
    std::vector<rclcpp::Parameter> parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  std::lock_guard<std::mutex> lock_reinit(mutex_);

  for (auto parameter : parameters) {
    const auto &type = parameter.get_type();
    const auto &name = parameter.get_name();

    if (type == ParameterType::PARAMETER_DOUBLE) {
      if (name == plugin_name_ + ".moving_kp") {
        moving_kp_ = parameter.as_double();
      } else if (name == plugin_name_ + ".moving_ki") {
        moving_ki_ = parameter.as_double();
      } else if (name == plugin_name_ + ".moving_kd") {
        moving_kd_ = parameter.as_double();
      } else if (name == plugin_name_ + ".heading_kp") {
        heading_kp_ = parameter.as_double();
      } else if (name == plugin_name_ + ".heading_ki") {
        heading_ki_ = parameter.as_double();
      } else if (name == plugin_name_ + ".heading_kd") {
        heading_kd_ = parameter.as_double();
      } else if (name == plugin_name_ + ".transform_tolerance") {
        double transform_tolerance = parameter.as_double();
        transform_tolerance_ = tf2::durationFromSec(transform_tolerance);
      } else if (name == plugin_name_ + ".min_max_sum_error") {
        min_max_sum_error_ = parameter.as_double();
      } else if (name == plugin_name_ + ".lookahead_distance") {
        lookahead_distance_ = parameter.as_double();
      }
    }
  }
  result.successful = true;
  return result;
}

}  // namespace nav2_pure_pursuit_controller

// Register this controller as a nav2_core plugin
PLUGINLIB_EXPORT_CLASS(nav2_pure_pursuit_controller::PurePursuitController,
                       nav2_core::Controller)
