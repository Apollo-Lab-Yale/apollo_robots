import pyrealsense2 as rs
import numpy as np
import cv2

class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # Filters
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 5)

        self.threshold = rs.threshold_filter(min_dist=0.2, max_dist=2.0)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.3)
        self.temporal.set_option(rs.option.filter_smooth_delta, 30)
        self.temporal.set_option(rs.option.holes_fill, 3)

        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 1)

        # Warm-up
        for _ in range(60):
            self.pipeline.wait_for_frames()

        # Cache depth scale
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        print("Camera initialized and warmed up.")

    def capture_aligned_rgbd(self, save: bool = True):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Apply RealSense filter chain
        depth_frame = self.threshold.process(depth_frame)

        # Work in disparity domain for better filtering
        depth_frame = self.depth_to_disparity.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)

        # Final cleanup
        depth_frame = self.hole_filling.process(depth_frame)

        # Convert to arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_meters = depth_image.astype(np.float32) * self.depth_scale

        if save:
            cv2.imwrite("rgb.png", color_image)
            np.save("depth.npy", depth_meters)

        return color_image, depth_meters

    def stop(self):
        self.pipeline.stop()