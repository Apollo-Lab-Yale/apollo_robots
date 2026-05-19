import zmq
import cv2
import pyrealsense2 as rs
import numpy as np

def main():
    # ---------- ZeroMQ -----------
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://127.0.0.1:6000")
    print("Server: Ready, streaming frames...")

    # ---------- RealSense Setup ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert to numpy
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())  # 16-bit depth

            # Encode RGB → JPEG
            _, color_encoded = cv2.imencode(".jpg", color_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            color_bytes = color_encoded.tobytes()

            # Encode depth → PNG (lossless, keeps 16-bit)
            _, depth_encoded = cv2.imencode(".png", depth_img)
            depth_bytes = depth_encoded.tobytes()

            # Send multipart: [RGB, DEPTH]
            socket.send_multipart([color_bytes, depth_bytes])

    except KeyboardInterrupt:
        print("Stopping server...")

    finally:
        pipeline.stop()
        socket.close()
        context.term()

if __name__ == "__main__":
    main()

