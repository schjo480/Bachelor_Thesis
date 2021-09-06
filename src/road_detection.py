from file_utils import *
from lidar_utils import *
from image_utils import *

################################################################################
#
# Title: Fahrspurerkennung.py
# Authors: Girstl, Felix
# Date: 2018
#
################################################################################


# Logic Operation
def logic_and(img1, img2, img3=None, img4=None):
    """
    Performs a logic AND with 2 to 4 images of the same size.
    The images mustn't have more than one channel.
    Usually Binary images are used.
    Output is a single one channel image with 0 or 1 as values.
    So only if the pixel value is true (>0) in every of the input images, the output image pixel will be true.
    """

    # set up image of the same size
    binary = np.zeros_like(img1)

    if type(img3) is not np.ndarray and type(img4) is not np.ndarray:  # only img1 and img2 are there as input
        # compare pixel values ( >0 ??? )
        binary[(img1 > 0) & (img2 > 0)] = 255

    elif type(img4) is not np.ndarray:  # img1/2&3 are there as input
        binary[(img1 > 0) & (img2 > 0) & (img3 > 0)] = 255

    elif type(img1) is np.ndarray and \
                    type(img2) is np.ndarray and \
                    type(img3) is np.ndarray and \
                    type(img4) is np.ndarray:  # there are 4 image inputs
        binary[(img1 > 0) & (img2 > 0) & (img3 > 0) & (img4 > 0)] = 255

    else:  # wrong input
        print('WRONG INPUT! CHECK THE IMAGE INPUT!')

    return binary


def logic_or(img1, img2, img3=None, img4=None):
    """
    Performs a logic OR with 2 to 4 images of the same size.
    The images mustn't have more than one channel.
    Usually Binary images are used.
    Output is a single one channel image with 0 or 1 as values.
    So only if the pixelvalue is true (>0) in at least one of the input images, the output image pixel will be true.
    """
    # set up image of the same size
    binary = np.zeros_like(img1)

    if type(img3) is not np.ndarray and type(img4) is not np.ndarray:  # only img1 and img2 are there as input
        # compare pixel values ( >0 ??? )
        binary[(img1 > 0) | (img2 > 0)] = 255

    elif type(img4) is not np.ndarray:  # img1/2&3 are there as input
        binary[(img1 > 0) | (img2 > 0) | (img3 > 0)] = 255

    elif type(img1) is np.ndarray and \
                    type(img2) is np.ndarray and \
                    type(img3) is np.ndarray and \
                    type(img4) is np.ndarray:  # there are 4 image inputs
        binary[(img1 > 0) | (img2 > 0) | (img3 > 0) | (img4 > 0)] = 255

    else:  # wrong input
        print('WRONG INPUT! CHECK THE IMAGE INPUT!')

    return binary


def det_line_steps(img1, img2, step=1):
    """
    Detects lines by canny and line image comparison (if canny-edge and next to it a colored line ->detected).
    Checks only certain image rows.
    :param img1: Canny image 1channel
    :param img2: line image 1channel
    :param step: distance between the checked rows
    :return: semi-binary image (pixelvalues of 0 or 255) 1channel
    """
    binary = np.zeros((img1.shape[0], img1.shape[1], 1), np.uint8)
    logic_and = np.zeros_like(img1)
    logic_and[(img1 > 0) & (img2 > 0)] = 255
    for i in range(0, img1.shape[0], step):  # certain rows

        j = 0  # initialization of j

        # draw red lines to show wich lines where checked
        cv2.line(binary, (0, i), (binary.shape[1], i), 0)

        # check if pixel values are greater than 0 in both pictures
        while j < img1.shape[1] - 1:  # columns
            if img1[i, j] > 0 and img2[i, j + 1] > 0 and logic_and[i, j] > 0:
                binary[i, j, :] = 255
                j += 0  # nahe doppelwertungen verhindern durch erhöhung des spaltenindices
            elif img1[i, j] > 0 and img2[i, j - 1] > 0 and logic_and[i, j] > 0:  # rechte kante der linie
                binary[i, j, :] = 255
            j += 1
    return binary


def det_line_1lane_init(canny, lineimg1, lineimg2, dataset, step=1):
    """
    Detects lines on moving stripes (more than one striperow with 2 lines to detect)
    The stripes have a various width (linear from max and min width over y)->inside the stripe list (2nd input of every
    row).
    Initialises the starting position of the stripes.(not manually set)
    :param canny: result of canny_edge()
    :param lineimg1: result of color filter (e.g. white)
    :param lineimg2: result of alternative color filter (e.g. yellow)
    :param dataset:
    :param step:
    :return:
    """
    output = np.zeros((canny.shape[0], canny.shape[1], 3), np.uint8)

    def frame_counter():
        '''Defines the frame counter if not existing already. And increases it every frame.'''
        if 'frame_nbr' not in globals():  # frame number counter / definition
            global frame_nbr
            frame_nbr = 1
        else:
            frame_nbr += 1
        return

    # "points" list's handling function
    def enter_new_midpoint(value, spot, nmax=10): #maybe decrease nmax from the original 25
        """
        Enters a new midpoint(value) into the list.
        Also shifts the old points to the "right" -> so the newest point is in the first listplace.
        :param value: value that shall be entered as the last detected midpoint
        :param spot: spot is the points[x][1-z]
        :param nmax: maximum of the allowed spaces for former midpoints
        :return:
        """

        # append a new space if less than the allowed spaces exist until sufficient
        while len(spot) < nmax:
            spot.append(None)

        # shift old points one to the right
        for i in range(nmax - 1, 0, -1):  # counts down from the last space index to 1
            spot[i] = spot[i - 1]  # shift the value to the space one on the right

        spot[0] = value

        return

    def get_lastvalid_midpoint(spot):
        '''
        returns the last value of the midpoints that is not None.
        spot is the points[x][1-z] and asks for a special midpoint
        :param self:
        :return:
        '''
        for point in spot:
            if point is not None:
                break

        return point

    def setup():
        """
        Sets up all lists
        :return:
        """

        def stripe_width(y, ymax=canny.shape[0], wmin=50, wmax=120):
            """
            calculates the linear width of the stripes. Depending on the row hight y.
            Smaller width on the top of the image.
            -> width = m*y+t m=(wmax-wmin)/ymax  t=wmin ymax=img.shape[1]
            :param y:
            :param wmin: minimum width of stripes
            :param wmax: maximum width
            :return: width of the row
            """
            if dataset == "KITTI":
                wmin = 30
                wmax = 380
            elif dataset == "Apolloscape_stereo":
                wmin = 140
                wmax = 1100
            elif dataset == "Oxford":
                wmin = 60
                wmax = 700
            width = int(((wmax - wmin) / ymax) * y + wmin)
            return width

        if "stripe" not in globals():
            global stripe  # list for checking stripe's midpoints
            stripe = []
            global points  # list for detected lane markings points (None if not detected)
            points = []

            global det_counter  # list which has info if stripe ever detected a point (True/False)
            det_counter = []
            if dataset=="Apolloscape_stereo":
                mid = 2.25
            else:
                mid = 2.0
            # set stripe default values for all rows
            for row in range(0, canny.shape[0] - 1, 1):  # iteration through rows
                stripe_mid = int(canny.shape[1] / mid)# stripes are in the middle of the width (going to move outwards)
                width = stripe_width(row)
                stripe.append([row, width, stripe_mid, stripe_mid])  # stripe = [[row, width, x_left, x_right], ...]

                det_counter.append([None, None])  # set the counter to None

                points.append([row, [], []])  # points = [[row, [last points left], [last points right]], ...]
                # set last values of points to none
                for i in range(1, 3):  # left and right iteration
                    enter_new_midpoint(None, points[int(row / 1)][i])

    def draw_stripes():
        # def output image (NOT NECESSARY ONLY FOR DEVELOPMENT AND VISUALISATION)
        output = np.zeros((canny.shape[0], canny.shape[1], 3), np.uint8)

        for row in stripe:
            y = row[0]
            w = row[1]
            mid_left = row[2]
            mid_right = row[3]
            output[y, int(mid_left - w / 2):int(mid_left + w / 2), 2] = 255
            output[y, int(mid_right - w / 2):int(mid_right + w / 2), 2] = 255
            output[y, int(mid_left - w / 2):int(mid_left + w / 2), 0] = 255
            output[y, int(mid_right - w / 2):int(mid_right + w / 2), 0] = 255

        return output

    def outmoving_stripe(pixelspeed=4): #decreased pixel speed from 9 to 4
        '''Moves the stripes to the left / right until they have found a marking.'''
        for i, row in enumerate(stripe):  # iterate through all stripes
            if get_lastvalid_midpoint(points[i][1]) is None:  # if point is None(not detected)->move left stripe to
                # the left
                if det_counter[i][0] is None:  # move only if stripe never detected something
                    stripe[i][2] -= pixelspeed

            if get_lastvalid_midpoint(points[i][2]) is None:  # right point(right lane marking)
                if det_counter[i][1] is None:
                    stripe[i][3] += pixelspeed

        return

    def end_outmoving_stripe():
        """
        Sets all stripes counter to True that have a lower and higher stripe that already detected something
        :return:
        """
        # get highest and lowest stripe that already detected something for both sides
        global end_initialisation_framenbr
        end_initialisation_framenbr = 3 #Maybe change this!! Original 53

        highest_left = None
        lowest_left = None
        highest_right = None
        lowest_right = None
        for i, info in enumerate(det_counter):
            if info[0]==True:  # if left stripe in this row ever detected anything
                if lowest_left is None:
                    lowest_left = i
                highest_left = i
            if info[1]==True:  # right stripe
                if lowest_right is None:
                    lowest_right = i
                highest_right = i
        # set all spaces of det_counter to True between the highest and lowest already detected points
        if lowest_right and highest_right and lowest_left and highest_left:
            for i in range(lowest_left, highest_left + 1):
                det_counter[i][0] = True
            for i in range(lowest_right, highest_right + 1):
                det_counter[i][1] = True

        # set all the det_counter values to True after specific amount of frames
        if frame_nbr == end_initialisation_framenbr:  # canny.shape[1]/8/2:
            for i, row in enumerate(det_counter):  # all rows
                for it in range(0, 2):  # left and right
                    det_counter[i][it] = True
        return

    def runnaway_stripe(maxdist=20): #changed maxdist from originally 100 to 10
        if frame_nbr > end_initialisation_framenbr:  # check for wrong stripes after initialisation
            # iterate all rows of stripe

            # stripe position from top to bottom
            for i in range(len(stripe) - 1):
                for it in range(2, 4): #left and right
                    if abs(stripe[i][it] - stripe[i + 1][it]) > maxdist and abs(stripe[i][it] - stripe[i - 1][it]) < \
                            maxdist:
                        # set stripe above to same value
                        stripe[i + 1][it] = stripe[i][it]

            # is the stripe too far to the left or too far to the right (focus on bottom half, because less curvature
            # has occurred here from bottom to middle
            for i in range(len(stripe) - 2, int(len(stripe)/2), -1):
                for it in range(2, 4): #left and right
                    if stripe[i][2] < 300:
                        stripe[i][2] = stripe[i + 1][2]
                    if stripe[i][3] > (canny.shape[1] - 300):
                        stripe[i][3] = stripe[i + 1][3]

        return

    def detector():
        lines = lineimg1 + lineimg2
        for idx1, entry in enumerate(stripe):  # iterate over all rows (idx1 ist der Index des Eintrags in stripe)

            # Stripes: draw and calculate
            for idx2, mittelpkt in enumerate(entry[2:]):  # iterate over all midpoints(stripes) of the row
                # (therefore skip y and w)

                # calculate stripe points
                pkt_l = int(mittelpkt - 0.5 * entry[1])  # left/right point of the stripe (entry[1] = width of stripe)
                pkt_r = int(mittelpkt + 0.5 * entry[1])

                # limitation of the stripe position (has to be in the image boundaries)
                if pkt_r >= lineimg1.shape[1]:  # if the stripe points are out of the image set to left or right
                    # max position
                    pkt_r = lineimg1.shape[1] - 1
                    pkt_l = pkt_r - entry[1]
                elif pkt_l < 0:
                    pkt_l = 0
                    pkt_r = entry[1]

                y = entry[0]

                # draw stripe (visualisation)
                # cv2.line(output, (pkt_l, y), (pkt_r, y), (0, 0, 255))  # entry[0]=y

                # checking
                list_hits = []  # list for all the hits for one loop
                # check all pixels on the stripe for lane marking points
                for x in range(pkt_l, np.min([pkt_r, canny.shape[1] - 2])):  # iterate over x
                    if x < canny.shape[1]:
                        if canny[y, x] > 0 and (lines[y, x + 1] > 0 or lines[y, x - 1] > 0):  # check left & right edge
                            # output[y, x, :] = 255
                            list_hits.append(x)  # append new hit to list

                # calculate new midpoint & save in stripe list
                if len(list_hits):  # if entries are in list -> else division thru 0
                    midpoint = int(sum(list_hits) / len(list_hits))
                    output[y, midpoint, :] = 255
                    stripe[idx1][idx2 + 2] = midpoint  # (idx2+2 ,because first 2 entries of stripe entries are
                    # y & width)
                    # detected -> put point into points list
                    enter_new_midpoint(midpoint, points[idx1][idx2 + 1])

                    # set the detection counter to True (stops the moving process of the stripe)
                    det_counter[idx1][idx2] = True
                else:  # if no point on stripe detected set points value to None

                    enter_new_midpoint(None, points[idx1][idx2 + 1])

    # Call all functions
    frame_counter()  # increase frame nbr
    setup()  # setup the lists...
    output = draw_stripes()
    detector()
    outmoving_stripe()
    end_outmoving_stripe()
    runnaway_stripe()

    return output