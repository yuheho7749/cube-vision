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
        "green": ((40, 40, 40), (80, 255, 255)),
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

    # Cleanup filtering
    final_output, smoothed_mask = smooth_filter(combined_output, combined_mask)
    edges = edge_detection(image, smoothed_mask)

    # Show the combined output image
    # cv2.imshow('Final Output', final_output)
    # cv2.imshow('Final mask', smoothed_mask)
    # cv2.imshow('edges', edges)
    # return final_output, smoothed_mask, edges
    return smoothed_mask


def main():
    video=cv2.VideoWriter('output-video-masks.mp4',cv2.VideoWriter_fourcc(*"mp4v"),30,(720, 1280), False)
    cam = cv2.VideoCapture("test-cube.mp4")
    ret, frame = cam.read()
    while ret:
        # cv2.imshow('video', frame)
        outputs = process_image(frame)
        # print(outputs.shape)
        video.write(outputs)
        ret, frame = cam.read()
            
    # cv2.destroyAllWindows()
    video.release()
    print("Done")




if __name__ == '__main__':
    main()

