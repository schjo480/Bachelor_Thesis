from road_detection import *

def overlay_road_on_image(img, dataset):
    """

    :param img: RGB image
    :param dataset:
    :return: RGB image with the detected road by det_line_1lane_init() overlaid on top of it.
    """
    output = img
    roi = roi_r(img, dataset)
    roi = gauss(roi, dataset)
    canny_edge = canny(roi)
    white_line = filter_white_hls_binary(roi)
    yellow_line = filter_yellow_hls_binary(roi)
    semantic_image = det_line_1lane_init(canny_edge, white_line, yellow_line, dataset)

    if dataset=="KITTI":
        roi_factor_y = 0.45
        roi_factor_x = 0.3
    elif dataset == "Apolloscape_stereo":
        roi_factor_y = 0.39
        roi_factor_x = 0.3
    elif dataset=="Oxford":
        roi_factor_y = 0.45
        roi_factor_x = 0.3

    factor_y = int(roi_factor_y * img.shape[0])
    factor_x = int(roi_factor_x * img.shape[1])

    output[factor_y:, factor_x:output.shape[1]-factor_x, :] = 0.6 * img[factor_y:, factor_x:img.shape[1]-factor_x, :] +\
                                                              0.4 * semantic_image

    return output


def get_road_category(longitude, latitude):
    """
    The api takes the format: longitude, latitude in decimal format.
    Connection to TUM network required. (e.g. by using the cisco anyconnect VPN)
    :param longitude:
    :param latitude:
    :return: road category and type as defined by OSM
    """
    url = "http://gis.ftm.mw.tum.de/reverse?coordinates=[" + str(longitude) + "," + str(latitude) + "]"
    response = urlopen(url)
    data = json.loads(response.read())
    #addresstype = data['features'][0]['properties']['addresstype']
    category = data['features'][0]['properties']['category']
    location = data['features'][0]['properties']['display_name']
    type = data['features'][0]['properties']["type"]

    return category, location, type
