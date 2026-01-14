import cv2
import numpy as np

def load_and_resize(path, scale=0.5):
    img_fullscale = cv2.imread(path)
    dsize = (int(img_fullscale.shape[1] * scale), int(img_fullscale.shape[0] * scale))
    return cv2.resize(img_fullscale, dsize)

def detect_features(img):
    orb = cv2.ORB_create()
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(img_bw, None)

    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints, descriptors

def match_features(img1, kp1, des1, img2, kp2, des2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    MATCH_N = 20

    matches = sorted(matches, key=lambda x: x.distance)

    final_img = cv2.drawMatches(img1, kp1, 
                             img2, kp2, matches[:MATCH_N], None)

    cv2.imshow("Matches", final_img)
    cv2.waitKey(0)

    return matches[:MATCH_N]

def extract_matched_points(kp1, kp2, matches):
    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)
        
    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)
        
    # Reshape the array to OpenCV-required format (N, 1, 2)

    print("First image matched points: ", pts1)
    print("Second image matched points: ", pts2)

    return pts1, pts2


def estimate_homography(pts1, pts2):
    H, mask = cv2.findHomography( # Estimate perspective transform
        pts2,                    # Source points (img2)
        pts1,                    # Destination points (img1)
        cv2.RANSAC,              # Use RANSAC to reject outliers
        5.0                      # Maximum reprojection error threshold
    )
    print(H)
    return H                     # Return 3×3 homography matrix


def warp_image(img, H, output_size):
    cv2.imshow("Original image", img)
    cv2.waitKey(0)                # Wait indefinitely for a key press

    # Use cv2.warpPerspective() to warp the image based on H
    warped_image = cv2.warpPerspective(img, H, output_size)

    cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)

    return warped_image


def blend_images(img1, warped_img2):
    panorama = warped_img2.copy()
    h, w = img1.shape[:2]
    panorama[:h, :w] = img1
    return panorama


def stitch_images(img1_path, img2_path):
    img1 = load_and_resize(       # Load and downscale first image
        img1_path
    )

    img2 = load_and_resize(       # Load and downscale second image
        img2_path
    )

    kp1, d1 = detect_features(img1)
    kp2, d2 = detect_features(img2)

    matches = match_features(img1, kp1, d1, img2, kp2, d2)

    pts1, pts2 = extract_matched_points(kp1, kp2, matches)

    H = estimate_homography(pts1, pts2)

    h, w = img1.shape[:2]
    warped_img = warp_image(img2, H, (w * 2,h))

    return blend_images(img1, warped_img)

if __name__ == "__main__":
    pano = stitch_images(         # Run panorama stitching pipeline
        "dresser_left.JPG",
        "dresser_right.JPG"
    )
    cv2.imshow(                   # Display resulting panorama
        "Panorama",
        pano
    )
    cv2.waitKey(0)                # Wait indefinitely for a key press
    cv2.destroyAllWindows()       # Close all OpenCV windows

