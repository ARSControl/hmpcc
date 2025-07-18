# teleop_actor.py (example)
import rospy, math
from geometry_msgs.msg import PoseStamped

rospy.init_node("actor_teleop")
pub = rospy.Publisher("/actor_pose", PoseStamped, queue_size=10)
rate = rospy.Rate(5)                      # 5 Hz

t0 = rospy.get_time()
while not rospy.is_shutdown():
    t = rospy.get_time() - t0
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = 2.0*math.sin(0.2*t)
    msg.pose.position.y = 2.0*math.cos(0.2*t)
    msg.pose.position.z = 1.0
    msg.pose.orientation.w = 1.0          # facing forward
    pub.publish(msg)
    rate.sleep()
