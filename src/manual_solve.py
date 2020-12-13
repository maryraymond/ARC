#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

# Student Name: Mary Raymond
# Student ID: 20235859
# Git hub repo: https://github.com/maryraymond/ARC.git

# Summary/reflection on the communality of solving the tasks
# Note that through out this code the task grids will be referred as images,
# and the squares in the grid will be referred to as pixels
# First data representation:
# - Each square in the grid (called here pixel) is represented as a dictionary of color, i and j
# - Each continuous colored pixels (continuous from up, down, left and right not diagonal) is called here
#   an object, and is represented as a list of pixels
# Second commonly used helper functions:
# - get_colored_pixels: this function filters the colored pixels from the background pixels and it is used
#   in all the three tasks as a staring point
# - get_object_list: this is one of the most useful functions internally it calls a number of helper functions
#   to break down the given list of colored pixels into objects, and this is very useful to identify the different
#   shapes in the image which is usually one step of solving the task
# - get_boundary: this function returns the boundary of a given object in terms of i_min, i_max (left and right loc)
#   and j_min, j_max (top and bottom loc) these information is useful for all the tasks to find the loc and dimension
#   and object
# - np.fliplr: this function is very useful when the image required to be flipped
# - get_neighbour_pixels: this function uses recursion to find all the adjacent pixels that forms an object,
#   staring from a given top left pixel, it is called from get_object_list to track each pixel and return the
#   complete object of which this pixel is part of
# Third thoughts about solving the ARC using AI
#   As observed by using the manual methods to solve a number of problems it seems clear that one important step in
#   solving the task would be to identify all the objects in the grid/image and starting from there based on the
#   objects color, location and / or shape drive the required transformation, so this would mean that whatever
#   AI algorithm that attempt to solve the ARC needs to know how to extract those objects and their information

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.


def solve_846bdb03(x):
    """ the required transformation for the task number 846bdb03 is as follows:
        1- identify the side bars (they have the top and bottom pixel color as yellow.
        2- find the object/shape in the image and check if it's left color matches the
           color of the left bar if not flip the object left to right to have the correct color
           at the correct side
        3- move the object to be in between of the 2 bars level the line of the top and bottom
           yellow pixel
        4- The output will be like a crop of the bars with the object in between so the output size
           will depend of the length of the side bars and the distance between them

        For this task all the test and training grids are solved correctly
        """
    # first get the colored pixels
    colored_pixels = get_colored_pixels(x)
    # Then we will get the objects list of the objects contained in the image
    object_list = get_object_list(x, colored_pixels)

    # Now we will search for the left and right bars in the object list we have
    # we also want to get the color of each bar
    left_bar, right_bar, left_color, right_color = get_left_right_obj(x, object_list)
    # remove the left and right bars from the list of colored pixels
    # so that we are left wit the remaining middle object
    colored_pixels = subtract_object(left_bar, colored_pixels)
    colored_pixels = subtract_object(right_bar, colored_pixels)

    # Now that we have the side bars we will check which
    # we will use it to determine the width and height of the
    # output image
    left_i_max, left_i_min, left_j_max, left_j_min = get_boundary(left_bar)
    right_i_max, right_i_min, right_j_max, right_j_min = get_boundary(right_bar)
    # The length of both objects should be the same
    assert (left_j_max - left_j_min) == (right_j_max - right_j_min)
    output_height = left_j_max - left_j_min + 1
    output_width = right_i_max - left_i_min + 1
    # find the offset that we need to abject the objects location with,
    # the top left of the left bar should be located at 0, 0
    i_offset = left_i_min - 0
    j_offset = left_j_min - 0
    # offset the left and right bars to their location in the output image
    offset_objects([left_bar, right_bar], i_offset, j_offset)
    output = np.zeros((output_height, output_width))
    # Call the draw function for the 2 bars
    [draw_objects(output, obj) for obj in [left_bar, right_bar]]

    # now we will check the middle object
    i_max, i_min, j_max, j_min = get_boundary(colored_pixels)
    # first we will need to check the colors to see if we need to flip it
    left_pixels_color = [pixel['color'] for pixel in colored_pixels if pixel['i'] == i_min]
    # find the offset of the object given that it should start at 1, 1
    i_offset = i_min - 1
    j_offset = j_min - 1
    # offset the middle object
    offset_objects([colored_pixels], i_offset, j_offset)
    # create an overlay image for the middle object
    output_overlay = np.zeros((output_height, output_width))
    # Draw the middle object to the overlay, so that we move the middle object
    # to its new location in the output image, this step is required before we
    # draw the object directly to the output image because we might need to flip
    # the object to match the left and right colors
    draw_objects(output_overlay, colored_pixels)

    # Check if we got the correct color in the left side
    if left_pixels_color[0] != left_color:
        # flip the middle object
        output_overlay = np.fliplr(output_overlay)
    # overlay the 2 parts of the output
    output += output_overlay

    return output


def solve_3befdf3e(x):
    """ the required transformation for the task number 846bdb03 is as follows:
        1- Find the squares in the grid / image each square has of 2 colors
        2- Swap the colors of the pixels of the square
        3- for each side of the square add an additional border line with thickness
           equal to have the square width ( if odd size then take the floor)
        4- the color of the additional boarder should be the same as the new color of
           the square core

        For this task all the test and training grids are solved correctly
    """
    # first we will find the colored
    colored_pixels = get_colored_pixels(x)
    # Initially the output is a copy of the input and then we will modify it
    output = np.copy(x)
    # get the color counts to determine the major and minor color
    color_count = get_colors_count(colored_pixels)
    major_color = int(max(color_count, key=color_count.get))
    minor_color = int(min(color_count, key=color_count.get))
    # Switch the two colors
    switch_colors(x, output, minor_color, major_color)
    # now we need to get a list of objects in the image
    object_list = get_object_list(x, colored_pixels)
    for curr_object in object_list:
        # Get the boundaries of the square object
        i_max, i_min, j_max, j_min = get_boundary(curr_object)
        # Get the width of the square and use it to calculate
        # the thickness of the added boundary
        boundary_thickness = int((i_max + 1 - i_min) / 2)
        # Add the additional boundary lines with the specified thickness and color
        add_boundary(output, (i_min, i_max, j_min, j_max), boundary_thickness, major_color)
    return output


def solve_67385a82(x):
    """ the required transformation for the task number 846bdb03 is as follows:
            1- The objects in the image
            2- for all the objects that are have more than one pixel color them blue
            3- the output should be the same as the input except with the objects
            bigger than 1 pixel colored in blue

            For this task all the test and training grids are solved correctly
        """
    # First we will just copy the th input to the output to start from there
    output = np.copy(x)
    # second we will get the colored pixels in the image
    colored_pixels = get_colored_pixels(x)
    # Then we will get the objects list of the objects contained in the image
    object_list = get_object_list(x, colored_pixels)
    # now we will create a list of objects that have more than one pixel
    large_objects = [curr_object for curr_object in object_list if len(curr_object) > 1]
    # Now we will color the object as blue
    blue = 8
    [color_object(obj, blue) for obj in large_objects]
    # Now we will draw the large objects with the new color
    [draw_objects(output, obj) for obj in large_objects]

    return output


def solve_445eab21(x):
    """ the required transformation for the task number 846bdb03 is as follows:
        1- find the rectangles in the grid / image and calculate the area of each
        2- the color os the output is the same as the rectangle with the largest area
        3- the output grid / image is always 2x2

        For this task all the test and training grids are solved correctly
    """
    # Initially our maximum area is 0 and the output color is black
    max_area = 0
    output_color = 0
    # First we will get the colored pixels in the image
    colored_pixels = get_colored_pixels(x)
    # Then we will get the objects list of the objects contained in the image
    object_list = get_object_list(x, colored_pixels)
    # for each object we will get the width and height
    for curr_object in object_list:
        # since we now that each object is a rectangle then by finding the
        # object boundaries we can calculate it's width and height
        i_max, i_min, j_max, j_min = get_boundary(curr_object)
        width = (j_max - j_min) + 1
        height = (i_max - i_min) + 1
        # check if the area of this object is bigger than our current max area
        if width * height > max_area:
            # This is our new max area
            max_area = width * height
            # We will use th color of this object
            output_color = curr_object[0]['color']
    # Create the output image with the selected color
    output = np.ones((2, 2)) * output_color

    return output

## Following are the helper functions used in the above tasks solvers


def color_object(curr_object, color):
    for pixel in curr_object:
        pixel['color'] = color


def draw_objects(image, curr_object):
    for pixel in curr_object:
        image[pixel['j']][pixel['i']] = pixel['color']


def offset_objects(object_list, i_offset, j_offset):
    for curr_object in object_list:
        for pixel in curr_object:
            pixel['i'] -= i_offset
            pixel['j'] -= j_offset


def get_left_right_obj(image, object_list):
    # we want to search the objects to identify the 2 side lines
    # the distinct thing about them is that they start by a yellow pixel
    left_side = []
    right_side = []
    i_min = image.shape[1]
    i_max = 0
    left_color = 0
    right_color = 0

    for curr_object in object_list:
        if curr_object and curr_object[0]['color'] == 4:
            # check if this object is the left most object
            if curr_object[0]['i'] < i_min:
                # this is our current j_min
                i_min = curr_object[0]['i']
                # we will assign this object as the left object
                # but first copy the current left object to the right
                right_side = left_side
                left_side = curr_object
                left_color = curr_object[1]['color']
            # check if this object is the right most object
            if curr_object[0]['i'] > i_max:
                # this is our current j_min
                i_max = curr_object[0]['i']
                # we will assign this object as the left object
                # but first copy the current left object to the right
                left_side = right_side
                right_side = curr_object
                right_color = curr_object[1]['color']
    return left_side, right_side, left_color, right_color


def get_object_list(image, colored_pixels):
    object_list = []
    loc_colored_pixels = colored_pixels.copy()

    # while we still have colored pixels we will keep getting
    # objects
    while loc_colored_pixels:
        object_pixels = get_object(image, loc_colored_pixels)
        # Update the colored pixels list to remove the object pixels
        loc_colored_pixels = subtract_object(object_pixels, loc_colored_pixels)
        object_list.append(object_pixels)
    return object_list


def subtract_object(curr_object, colored_pixel):
    updated_colored_pixel = []
    for pixel in colored_pixel:
        if pixel not in curr_object:
            updated_colored_pixel.append(pixel)

    colored_pixel = updated_colored_pixel
    return colored_pixel


def get_object(image, colored_pixels):

    # check if the colored pixels list is not empty
    if colored_pixels:
        # we will get the top left pixel of the current list
        # of colored pixels and follow it to get the pixel object
        i_indexes = [object_data['i'] for object_data in colored_pixels]
        i_min = np.min(i_indexes)
        j_indexes = [object_data['j'] for object_data in colored_pixels  if object_data['i'] == i_min]
        j_min = np.min(j_indexes)
        top_left_pixel = {'color': image[j_min][i_min], 'i': i_min, 'j': j_min}
        # get the corresponding object
        object_pixels = get_continuous_object(image, top_left_pixel)
        # return the object pixels
        return object_pixels
    # if we got an empty color list we will just return


def get_continuous_object(image, top_left_pixel):
    object_pixels = []
    get_neighbour_pixels(object_pixels, image, top_left_pixel)
    # Get the unique values of the returned list
    unique_object_pixels = []
    for pixel in object_pixels:
        if pixel not in unique_object_pixels:
            unique_object_pixels.append(pixel)

    return unique_object_pixels


def get_neighbour_pixels(output_pixel, image, curr_pixel):

    # first we need to check if we have already checked this pixel
    # to avoid re-checking a pixel that we have already explored
    if curr_pixel not in output_pixel:
        # then we will add our curr pixel to the output pixel list
        output_pixel.append(curr_pixel)
        # for the current pixel we will check the 4 neighbours
        i = curr_pixel['i']
        j = curr_pixel['j']

        # check the lower pixel
        if j < image.shape[0] - 1:
            if image[j + 1][i] != 0:
                # call get neighbours for this pixel
                get_neighbour_pixels(output_pixel, image, {'color': image[j + 1][i], 'i': i, 'j': j+1})

        # check the upper pixel
        if j > 0:
            if image[j - 1][i] != 0:
                # call get neighbours for this pixel
                get_neighbour_pixels(output_pixel, image, {'color': image[j - 1][i], 'i': i, 'j': j - 1})

        # check the right pixel
        if i < image.shape[1] - 1:
            if image[j][i + 1] != 0:
                # call get neighbours for this pixel
                get_neighbour_pixels(output_pixel, image, {'color': image[j][i + 1], 'i': i + 1, 'j': j})

        # check the left pixel
        if i > 0:
            if image[j][i - 1] != 0:
                # call get neighbours for this pixel
                get_neighbour_pixels(output_pixel, image, {'color': image[j][i - 1], 'i': i - 1, 'j': j})

    # if no colored pixel was found on the neighbours we will return
    return


def add_boundary(image, boundary, thickness, color):
    i_start, i_end, j_start, j_end = boundary
    # for each side of the square we will add
    # the boundary with the required thickness
    # fill for the upper boundary
    j = j_start
    for i in range(i_start, i_end + 1):
        for offset in range(1, thickness + 1):
            image[j - offset][i] = color
    # fill for the lower boundary
    j = j_end
    for i in range(i_start, i_end + 1):
        for offset in range(1, thickness + 1):
            image[j + offset][i] = color

    # fill the left side boundaries
    i = i_start
    for j in range(j_start, j_end + 1):
        for offset in range(1, thickness + 1):
            image[j][i - offset] = color

    # fill the right side boundaries
    i = i_end
    for j in range(j_start, j_end + 1):
        for offset in range(1, thickness + 1):
            image[j][i + offset] = color


def get_boundary(curr_object):
    # get the list of the i index (width)
    i_indexes = [object_data['i'] for object_data in curr_object]
    # get the list of the j index (length)
    j_indexes = [object_data['j'] for object_data in curr_object]

    # get the max and min i index
    i_max = np.max(i_indexes)
    i_min = np.min(i_indexes)
    j_max = np.max(j_indexes)
    j_min = np.min(j_indexes)

    return i_max, i_min, j_max, j_min


def switch_colors(input, output, color1, color2):
    output[input[:, :] == color1] = color2
    output[input[:, :] == color2] = color1


def get_colored_pixels(x):
    objects = [{'color': color, 'i': i, 'j': j} for j, row in enumerate(x) for i, color in enumerate(row) if color != 0]
    return objects


def get_colors_count(objects):
    # find the majority and minority color
    color_set = {object_data['color'] for object_data in objects}
    color_count = dict()
    for color in color_set:
        color_count[str(color)] = len([object_data['color'] for object_data in objects if object_data['color'] == color])

    return color_count


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))


if __name__ == "__main__": main()

