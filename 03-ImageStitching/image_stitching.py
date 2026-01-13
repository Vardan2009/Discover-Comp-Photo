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

    cv2.imshow("Keypoints", img_with_keypoints)
    cv2.waitKey(0)                # Wait until a key is pressed

    return keypoints, descriptors


def match_features(img1, kp1, des1, img2, kp2, des2):
    # Create brute-force matcher using Hamming Distance, Use cv2.BFMatcher

    # Find the matches between image 1 and image 2 (Hint: use the .match() method)
    # matches = 

    # Sort the matches based on how close they are
    # matches = 

    # Draw the matches acorss both images (Hint: use the drawMatches() method)
    # drawMatches = cv2.drawMatches()

    cv2.imshow("Matches", drawMatches)
    cv2.waitKey(0)

    print(matches)
    return matches[:200]         # Keep only the best 200 matches to remove outliers


def extract_matched_points(kp1, kp2, matches):
    # Build array of matched points from image 1 and extract (x,y) coordinates of matched keypoints
    # pts1 = 
        
    # Reshape the array to OpenCV-required format (N, 1, 2)

    # Build array of matched points from image 2 and extract (x,y) coordinates of matched keypoints
    # pts2 = 
        
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
    # warped_image = cv2.warpPerspective()

    cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)

    return warped_image



def blend_images(img1, warped_img2):
    panorama = warped_img2.copy() # Start panorama as warped img2

    # Create mask of valid pixels in img1, value is True if at least one channel is nonzero in the pixel
    # mask = 

    # Use mask tooOverwrite panorama with valid image 1 data
    # panorama = 

    return panorama               # Return combined panorama image


def stitch_images(img1_path, img2_path):
    img1 = load_and_resize(       # Load and downscale first image
        img1_path
    )
    img2 = load_and_resize(       # Load and downscale second image
        img2_path
    )

    kp1, d1 = detect_features(img1)
    kp2, d2 = detect_features(img2)

    return img1

if __name__ == "__main__":
    pano = stitch_images(         # Run panorama stitching pipeline
        "dresser_left.jpg",
        "dresser_right.jpg"
    )
    cv2.imshow(                   # Display resulting panorama
        "Panorama",
        pano
    )
    cv2.waitKey(0)                # Wait indefinitely for a key press
    cv2.destroyAllWindows()       # Close all OpenCV windows

