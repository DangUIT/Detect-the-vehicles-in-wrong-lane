# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = []
        self.line_dist_thresh = 15
        self.counting_region = []
        self.region_color = (0, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = None  # Classes names
        self.annotator = None  # Annotator
        self.window_name = "DETECT THE VEHICLES IN WRONG LANE"

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = (255, 0, 0)
        self.count_bg_color = (255, 255, 255)
        self.cls_txtdisplay_gap = 50
        self.fontsize = 0.6

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = None

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        classes_names,
        reg_pts,
        count_reg_color=(0, 0, 255),
        count_txt_color=(0, 0, 255),
        count_bg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_color (RGB color): count text color value
            count_bg_color (RGB color): count highlighter line color
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
            cls_txtdisplay_gap (int): Display gap between each class count
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # Region and line selection
        if len(reg_pts[0]) == 2:
            print("Line Counter Initiated.")
            self.reg_pts.append(reg_pts[0])
            self.counting_region.append(LineString(self.reg_pts[0]))
        elif len(reg_pts[0]) >= 3:
            print("Polygon Counter Initiated.")
            for i in range(4):
                self.reg_pts.append(reg_pts[i])
                self.counting_region.append(Polygon(self.reg_pts[i]))
            # print(self.counting_region)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
            self.counting_region.append(LineString(self.reg_pts))

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        self.cls_txtdisplay_gap = cls_txtdisplay_gap

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters you may want to pass to the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts[0]):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region[0] = Polygon(self.reg_pts[0])

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        # Draw region or line
        for i in range(4):
            self.annotator.draw_region(reg_pts=self.reg_pts[i], color=self.region_color, thickness=self.region_thickness)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = {"SUM": 0}

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color if self.track_color else colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon
                if len(self.reg_pts[0]) >= 3:
                    is_inside = []
                    for i in range(4):
                        is_inside.append(self.counting_region[i].contains(Point(track_line[-1])))

                    if prev_position is not None and is_inside[0] and track_id not in self.count_ids:
                        self.count_ids.append(track_id)
                        if (box[1] - prev_position[1]) * (self.counting_region[0].centroid.x - prev_position[1]) < 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["SUM"] += 1
                    if is_inside[1]:
                        if self.names[cls] not in ['car', 'truck']:
                            self.annotator.box_label(box, label=f"wrong {self.names[cls]}#{track_id}",
                                                     color=colors(int(track_id), True))
                            f = open("Save/Data/lane1.txt", "a")
                            f.write(f"{self.names[cls]}#{track_id}\n")
                            f.close()
                    if is_inside[2]:
                        if self.names[cls] != 'bus':
                            self.annotator.box_label(box, label=f"wrong {self.names[cls]}#{track_id}",
                                                     color=colors(int(track_id), True))
                            f = open("Save/Data/lane2.txt", "a")
                            f.write(f"{self.names[cls]}#{track_id}\n")
                            f.close()
                    if is_inside[3]:
                        if self.names[cls] != 'motorbike':
                            self.annotator.box_label(box, label=f"wrong {self.names[cls]}#{track_id}",
                                                     color=colors(int(track_id), True))
                            f = open("Save/Data/lane3.txt", "a")
                            f.write(f"{self.names[cls]}#{track_id}\n")
                            f.close()
        labels_dict = {}

        for key, value in self.class_wise_count.items():
            if value["SUM"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    continue
                else:
                    labels_dict[str.capitalize(key)] = f"{value['SUM']}"

        # print(labels_dict)
        if labels_dict is not None:
            self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 1)

    def display_frames(self):
        """Display frame."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if len(self.reg_pts[0]) == 4:  # only add mouse event If user drawn region
                cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts[0]})
            cv2.imshow(self.window_name, self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):

        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        # count = 0
        # cv2.imwrite("Save/Picture/picture%d.jpg " % count, self.im0)

        if self.view_img:
            self.display_frames()
        return self.im0

