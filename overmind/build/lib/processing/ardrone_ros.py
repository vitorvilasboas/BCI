import rospy
import tf
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from time import sleep
from std_srvs.srv import Empty as srv_Empty
from math import pi

COMMAND_PERIOD = 10  # ms
Kp = 0.05

class ARDrone:
    def __init__(self):
        rospy.init_node('pilot', anonymous=True)
        self.pub_takeoff = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=1)
        self.pub_land = rospy.Publisher('/ardrone/land', Empty, queue_size=1)
        self.pub_reset = rospy.Publisher('/ardrone/reset', Empty, queue_size=1)

        # Gazebo Reset Simulator
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', srv_Empty)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.update_navdata)
        # self.pub_cmd = rospy.Publisher('/cmd_vel',Twist, 1)
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pos_x = 0
        self.pos_y = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.target_yaw = 270
        self.command = Twist()
        self.VelCommandTimer = rospy.Timer(rospy.Duration(COMMAND_PERIOD / 1000.0), self.send_cmd)
        rospy.on_shutdown(self.land)

    def takeoff(self):
        self.pub_takeoff.publish()

    def land(self):
        self.pub_land.publish()

    def stop(self):
        self.set_forward_vel(0)
        self.set_yaw_vel(0)

    def reset(self):
        self.pub_reset.publish()
        self.reset_world()
        self.target_yaw = 270

    def set_direction(self, direction):
        # print('setting new direction. old:', self.direction_list[self.direction_idx])
        yaw = self.target_yaw
        if direction == 'left': yaw = (yaw + 90.0) % 360
        elif direction == 'right': yaw = (yaw - 90.0) % 360
        self.target_yaw = yaw

    def lock_direction(self, pos):
        ref = self.target_yaw
        if (pos <= 90) and (ref >= 270): pos += 360
        error = ref - pos
        if abs(error) > 180: error = -error
        ctrl = Kp * error
        if ctrl > 1: ctrl = 1
        elif ctrl < -1: ctrl = -1
        self.set_yaw_vel(ctrl)

    def set_forward_vel(self, forward=0):
        # Called by the main program to set the current command
        self.command.linear.x = forward

    def set_yaw_vel(self, yaw=0):
        self.command.angular.z = yaw

    def send_cmd(self, event):
        # The previously set command is then sent out periodically if the drone is flying
        self.pub_vel.publish(self.command)

    def update_navdata(self, data):
        self.pos_x = data.pose[10].position.x
        self.pos_y = data.pose[10].position.y
        q_x = data.pose[10].orientation.x
        q_y = data.pose[10].orientation.y
        q_z = data.pose[10].orientation.z
        q_w = data.pose[10].orientation.w
        quaternion = (q_x, q_y, q_z, q_w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.roll = euler[0] * 180 / pi
        self.pitch = euler[1] * 180 / pi
        self.yaw = (euler[2] * 180 / pi) + 180  # offset to avoid - numbers
        self.lock_direction(self.yaw)

if __name__ == '__main__':
    d = ARDrone()
    # sleep(2)
    # drone.takeoff()
    # # sleep(10)
    # # drone.set_cmd(1, 0)
    # # drone.set_direction(1)
    # while True:
    #     # print drone.yaw
    #     # print drone.target_yaw
    #     drone.lock_direction(None)
    #     sleep(1)
    # drone.land()
