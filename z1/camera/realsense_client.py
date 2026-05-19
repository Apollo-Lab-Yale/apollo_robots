import zmq
import cv2
import numpy as np

def main():
    # ---------- ZeroMQ ----------
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.220:6000")
    print("Client: Waiting for frames...")

    while True:
        # Receive [RGB, DEPTH]
        color_bytes, depth_bytes = socket.recv_multipart()

        # Decode RGB
        color_arr = np.frombuffer(color_bytes, dtype=np.uint8)
        color_img = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)

        # Decode depth (16-bit PNG)
        depth_arr = np.frombuffer(depth_bytes, dtype=np.uint8)
        depth_img = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)  # Keeps uint16

        # Optional: visualize depth as colormap
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Show both
        cv2.imshow("RGB", color_img)
        cv2.imshow("Depth", depth_colormap)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

