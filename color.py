import cv2
import numpy as np

def smooth_filter(image, mask):
    kernel =  np.ones((10,10),np.uint8)
    # erode_filter = cv2.erode(combined_mask,kernel,iterations = 4)
    # cv2.imshow('erode', erode_filter)
    # dialated_filter = cv2.dilate(erode_filter,kernel,iterations = 4)
    # cv2.imshow('dialated', dialated_filter)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closing', closing)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opening', opening)

    blob_image = np.zeros_like(opening)
    for val in np.unique(opening)[1:]:
        mask = np.uint8(opening == val)
        labels, stats = cv2.connectedComponentsWithStats(opening, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        blob_image[labels == largest_label] = val

    final_output = cv2.bitwise_and(image, image, mask=blob_image)
    return final_output, blob_image

def edge_detection(image, mask):
    kernel =  np.ones((5,5),np.uint8)
    dialated_mask = cv2.dilate(mask,kernel,iterations = 1)

    edges = cv2.Canny(image,200,200)
    edges = cv2.bitwise_and(edges, edges, mask=dialated_mask)
    return edges

def process_image(image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    masks = {}
    outputs = {}
    combined_mask = np.zeros_like(hsv_image[:, :, 0])

    color_ranges = {
        "red": ((0, 190, 80), (5, 255, 255)),
        "orange": ((6, 210, 70), (20, 255, 255)),
        "green": ((50, 50, 50), (80, 255, 255)),
        "blue": ((100, 150, 0), (140, 255, 255)),
        "yellow": ((25, 150, 150), (35, 255, 255)),
        "white": ((0, 0, 200), (180, 20, 255))
    }



    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # Create a mask for the current color range
        mask = cv2.inRange(hsv_image, lower, upper)
        masks[color] = mask  # Store mask
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Create an output image that shows the original image where the mask is white
        output = cv2.bitwise_and(image, image, mask=mask)
        outputs[color] = output  # Store output


        # cv2.imshow(f"{color} mask", mask)
        # cv2.imshow(f"{color} output", output)


    combined_output = np.zeros_like(image)
    # Combine all output images
    for color, output in outputs.items():
        # Where output is not black (i.e., the color is detected), copy it to the combined_output image
        combined_output[output > 0] = output[output > 0]

    for color, mask in masks.items():
        combined_mask[mask > 0] = mask[mask > 0]

    # cv2.imshow('combined mask', combined_mask)

    # Cleanup filtering
    final_output, smoothed_mask = smooth_filter(combined_output, combined_mask)
    edges = edge_detection(image, smoothed_mask)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_image = cv2.bitwise_xor(image, image)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 3)
    kernel =  np.ones((3,3),np.uint8)
    contour_image = cv2.dilate(contour_image,kernel,iterations = 2)
    contour_image = cv2.morphologyEx(contour_image, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('contours', contour_image)

    tiles = cv2.bitwise_and(final_output, 255 - contour_image)
    # cv2.imshow('tiles', tiles)

    edges2 = cv2.Canny(tiles,200,200)
    # cv2.imshow('edge2', edges2)



    # Show the combined output image
    # cv2.imshow('Final Output', final_output)
    # cv2.imshow('Final mask', smoothed_mask)
    # cv2.imshow('edges', edges)
    return final_output, smoothed_mask, edges, contour_image, tiles, edges2
    # return smoothed_mask


def make_vid():
    video_output=cv2.VideoWriter('output/output2.mp4',cv2.VideoWriter_fourcc(*"mp4v"),30,(720, 1280), True)
    video_mask=cv2.VideoWriter('output/mask2.mp4',cv2.VideoWriter_fourcc(*"mp4v"),30,(720, 1280), False)
    video_edges=cv2.VideoWriter('output/edges2.mp4',cv2.VideoWriter_fourcc(*"mp4v"),30,(720, 1280), False)
    video_tile=cv2.VideoWriter('output/tiles2.mp4',cv2.VideoWriter_fourcc(*"mp4v"),30,(720, 1280), True)
    cam = cv2.VideoCapture("data/input-video.mp4")
    ret, frame = cam.read()
    while ret:
        # cv2.imshow('video', frame)
        outputs = process_image(frame)
        # print(outputs.shape)
        video_output.write(outputs[0])
        video_mask.write(outputs[1])
        video_edges.write(outputs[2])
        video_tile.write(outputs[4])
        ret, frame = cam.read()
            
    # cv2.destroyAllWindows()
    video_output.release()
    video_mask.release()
    video_edges.release()
    video_tile.release()
    print("Done")

def test_image():
    image = cv2.imread('test-cube.png') 
    output_image, smoothed_mask, edges, _, _, _ = process_image(image)

    cv2.imshow('edges', edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_image = cv2.bitwise_xor(image, image)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 3)
    kernel =  np.ones((3,3),np.uint8)
    contour_image = cv2.dilate(contour_image,kernel,iterations = 2)
    contour_image = cv2.morphologyEx(contour_image, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('contours', contour_image)

    tiles = cv2.bitwise_and(output_image, 255 - contour_image)
    cv2.imshow('tiles', tiles)

    edges2 = cv2.Canny(tiles,200,200)
    cv2.imshow('edge2', edges2)


    # contour_image2 = cv2.bitwise_xor(image, image)
    # contours2, hierarchy = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # rect = cv2.minAreaRect(tiles)
    # box = cv2.boxPoints(rect)
    # box = np.int32(box)
    # cv2.drawContours(image,[box],0,(0,0,255),2)
    # cv2.imshow('contours2', box)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    make_vid()
    # test_image()

