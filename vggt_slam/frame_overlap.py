import numpy as np
import cv2


def compute_image_sharpness(image):
    """Return variance-of-Laplacian sharpness for a BGR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FrameTracker:
    def __init__(self):
        self.last_kf = None
        self.kf_pts = None
        self.kf_gray = None

    def initialize_keyframe(self, image):
        self.last_kf = image
        self.kf_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.kf_pts = cv2.goodFeaturesToTrack(
            self.kf_gray,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )

    def accept_keyframe(self, image):
        """Commit an accepted image as the optical-flow reference keyframe."""
        self.initialize_keyframe(image)

    def compute_disparity_candidate(self, image, min_disparity, visualize=False):
        """Return whether ``image`` qualifies by disparity without changing state.

        Call :meth:`accept_keyframe` only after any additional keyframe gates
        have accepted this candidate. This keeps the reference state tied to
        the last accepted keyframe rather than the last proposed one.
        """
        if self.last_kf is None or self.kf_pts is None or len(self.kf_pts) < 10:
            return True

        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Track keyframe points into current frame
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.kf_gray, curr_gray, self.kf_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        status = status.flatten()
        good_kf = self.kf_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_kf) < 10:
            return True

        # Measure displacement from keyframe to current frame
        displacement = np.linalg.norm(good_next - good_kf, axis=1)
        mean_disparity = np.mean(displacement)

        if visualize:
            vis = image.copy()
            for p1, p2 in zip(good_kf, good_next):
                p1 = tuple(p1.ravel().astype(int))
                p2 = tuple(p2.ravel().astype(int))
                cv2.arrowedLine(vis, p1, p2, color=(0, 255, 0), thickness=1, tipLength=0.3)
            cv2.imshow("Optical Flow", vis)
            cv2.waitKey(1)

        if mean_disparity > min_disparity:
            return True
        return False

    def compute_disparity(self, image, min_disparity, visualize=False):
        """Legacy disparity-only selection API that commits qualifying frames."""
        is_candidate = self.compute_disparity_candidate(
            image, min_disparity, visualize
        )
        if is_candidate:
            self.accept_keyframe(image)
        return is_candidate
