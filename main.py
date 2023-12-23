import cv2
import numpy as np
import math
import matplotlib.pyplot as plt 
from prettytable import PrettyTable

def checkSymmetryX(image):
    if image.shape[0] % 2:
        image = cv2.copyMakeBorder(image, 0, 1, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)

    return (matchImages(image[0:image.shape[0]//2, :], cv2.flip(image[image.shape[0]//2:image.shape[0], :], 0)) / np.count_nonzero(image)) * 100

def checkSymmetryY(image):
    if image.shape[1] % 2:
        image = cv2.copyMakeBorder(image, 0, 0, 0, 1, cv2.BORDER_CONSTANT, None, value = 0)

    return (matchImages(image[:, 0:image.shape[1]//2], cv2.flip(image[:, image.shape[1]//2:image.shape[1]], 1)) / np.count_nonzero(image)) * 100

def crop(image):
    if len(image.shape) == 3:
         y_nonzero, x_nonzero, _ = np.nonzero(image)
    else:
        y_nonzero, x_nonzero = np.nonzero(image)

    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

def matchImages(image1, image2):
    difference_image = np.not_equal(image1, image2)
    
    if display_flag:
        cv2.imshow("Overlap", np.uint8(difference_image)*255)
        cv2.waitKey()
    
    return difference_image.sum()

def checkCircularity(image):
    image = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3) 

    if display_flag:
        cv2.imshow("Sobel Filter", image) 
        cv2.waitKey()

    center = [image.shape[0] // 2, image.shape[1] // 2]

    distances = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 255:
                distances.append(math.dist([i, j], center))

    return np.std(distances)

def getSharpness(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    if display_flag:
        cv2.imshow("Laplacian Filter", image) 
        cv2.waitKey()

    return np.var(laplacian) / 10

def getDiameter(image):
    return (np.count_nonzero(image) / (image.shape[0] * image.shape[1])) * 100

def colorVariation(image):
    if display_flag:
        cv2.imshow("Colored Lesion", image)
        cv2.waitKey()
    
        red_hist = cv2.calcHist([image], [2], None, [255], [1, 255])
        green_hist = cv2.calcHist([image], [1], None, [255], [1, 255])
        blue_hist = cv2.calcHist([image], [0], None, [255], [1, 255])
    
        plt.plot(red_hist, color='red')
        plt.xlim([1, 256])
        plt.plot(green_hist, color='green')
        plt.xlim([1, 256])
        plt.plot(blue_hist, color='blue')
        plt.xlim([1, 256])
        plt.show()

    b = np.std(image[:,:,0])
    g = np.std(image[:,:,1])
    r = np.std(image[:,:,2])

    return np.array([b, g, r])

def getProperties(filename):
    image = cv2.imread("PH2Dataset/PH2 Dataset images/" + filename + "/" + filename + "_lesion/" + filename + "_lesion.bmp", 0)
    color_image = cv2.imread("PH2Dataset/PH2 Dataset images/" + filename + "/" + filename + "_Dermoscopic_Image/" + filename + ".bmp", cv2.IMREAD_COLOR)

    for i in range(3):
        color_image[:,:,i] = (np.multiply(color_image[:,:,i], image//255))
    
    image = crop(image)
    cropped_color_image = crop(color_image) 

    return np.concatenate((np.array([checkSymmetryX(image), checkSymmetryY(image), checkCircularity(image), getDiameter(image), getSharpness(color_image)]), colorVariation(cropped_color_image)))

def distance(object_attributes, class_attributes):
    differences = object_attributes - class_attributes 
    return np.sqrt(np.sum(np.square(differences)))

def test_images(images, attributes):
    classes = ["Common  ", "Atypical", "Melanoma"]
    positives = 0
    negatives = 0

    classifications = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  #common, atypical, melanoma

    for i, type in enumerate(images):
        for image in type:
            temp = getProperties(image)
            #print(temp)
            distances = [distance(temp, attributes[0]), distance(temp, attributes[1]), distance(temp, attributes[2])]
            index = distances.index(min(distances))
            classifications[i][index] += 1
            print(image, "Diagnosis: {} True Value: {}".format(classes[index], classes[i])) 
            if index == i:
                positives += 1
            else:
                negatives += 1

    t = PrettyTable([" "] + classes)

    for i in range(3):
        temp = [classes[i]]
        for j in range(3):
            temp.append(classifications[i][j]) 
        t.add_row(temp)

    print(t)

    return positives, negatives, (positives / (positives + negatives)) * 100 

def getClassAttributes(images):
    attributes = []

    for type in images:
        temp = np.zeros((1, 8), dtype = np.float32)

        for image in type:
            #print(getProperties(image))
            temp += getProperties(image)

        temp = [x / len(type) for x in temp]
        attributes.append(temp)
    
    return attributes

def plotAttributes(attributes):
    
    labels = ["X Symmetry", "Y Symmetry", "Circularity", "Diameter", "Sharpness", "Blue", "Green", "Red"]
    length = len(attributes[0][0])
    x1 = np.arange(length) - 0.2
    x2 = np.arange(length)
    x3 = np.arange(length) + 0.2

    plt.scatter(x1, attributes[0], color='red', label='Common Nevus')
    plt.scatter(x2, attributes[1], color='green', label='Atypical Nevus')
    plt.scatter(x3, attributes[2], color='blue', label='Melanoma')

    plt.xlabel('Properties')
    plt.ylabel('Average Value')
    plt.xticks(range(len(labels)), labels)
    plt.title('Average Properties by Class')
    plt.legend()
    plt.show()


#common nevus, atypical nevus, melanoma
images = [['IMD003', 'IMD009', 'IMD016', 'IMD022', 'IMD024', 'IMD025', 'IMD035', 'IMD038', 'IMD042', 'IMD044', 'IMD045', 'IMD050', 'IMD092', 'IMD101', 'IMD103', 'IMD112', 'IMD118', 'IMD125', 'IMD132', 'IMD134', 'IMD135', 'IMD144', 'IMD146', 'IMD147', 'IMD150', 'IMD152', 'IMD156', 'IMD159', 'IMD161', 'IMD162', 'IMD175', 'IMD177', 'IMD182', 'IMD198', 'IMD200', 'IMD010', 'IMD017', 'IMD020', 'IMD039', 'IMD041', 'IMD105', 'IMD107', 'IMD108', 'IMD133', 'IMD142', 'IMD143', 'IMD160', 'IMD173', 'IMD176', 'IMD196', 'IMD197', 'IMD199', 'IMD203', 'IMD204', 'IMD206', 'IMD207', 'IMD208', 'IMD364', 'IMD365', 'IMD367', 'IMD371', 'IMD372', 'IMD374', 'IMD375', 'IMD378', 'IMD379', 'IMD380', 'IMD381', 'IMD383', 'IMD384', 'IMD385', 'IMD389', 'IMD390', 'IMD392', 'IMD394', 'IMD395', 'IMD397', 'IMD399', 'IMD400', 'IMD402']
          , ['IMD002', 'IMD004', 'IMD013', 'IMD015', 'IMD019', 'IMD021', 'IMD027', 'IMD030', 'IMD032', 'IMD033', 'IMD037', 'IMD040', 'IMD043', 'IMD047', 'IMD048', 'IMD049', 'IMD057', 'IMD075', 'IMD076', 'IMD078', 'IMD120', 'IMD126', 'IMD137', 'IMD138', 'IMD139', 'IMD140', 'IMD149', 'IMD153', 'IMD157', 'IMD164', 'IMD166', 'IMD169', 'IMD171', 'IMD210', 'IMD347', 'IMD155', 'IMD376', 'IMD006', 'IMD008', 'IMD014', 'IMD018', 'IMD023', 'IMD031', 'IMD036', 'IMD154', 'IMD170', 'IMD226', 'IMD243', 'IMD251', 'IMD254', 'IMD256', 'IMD278', 'IMD279', 'IMD280', 'IMD304', 'IMD305', 'IMD306', 'IMD312', 'IMD328', 'IMD331', 'IMD339', 'IMD356', 'IMD360', 'IMD368', 'IMD369', 'IMD370', 'IMD382', 'IMD386', 'IMD388', 'IMD393', 'IMD396', 'IMD398', 'IMD427', 'IMD430', 'IMD431', 'IMD432', 'IMD433', 'IMD434', 'IMD436', 'IMD437']
          , ["IMD058", "IMD061", "IMD063", "IMD064", "IMD065", "IMD080", "IMD085", "IMD088", "IMD090", "IMD091", "IMD168", "IMD211", "IMD219", "IMD240", "IMD242", "IMD284", "IMD285", "IMD348", "IMD349", "IMD403", "IMD404", "IMD405", "IMD407", "IMD408", "IMD409", "IMD410", "IMD413", "IMD417", "IMD418", "IMD419", "IMD406", "IMD411", "IMD420", "IMD421", "IMD423", "IMD424", "IMD425", "IMD426", "IMD429", "IMD435"]
         ]

display_flag = False
attributes = getClassAttributes(images)
plotAttributes(attributes)

print("Common Nevus, Atypical Nevus, Melanoma:")
for x in attributes:
    print("X: {}, Y: {}, Circularity: {}, Diameter: {}, Sharpness: {} B: {}, G: {}, R: {}".format(round(x[0][0]), round(x[0][1]), round(x[0][2]), round(x[0][3]), round(x[0][4]), round(x[0][5]), round(x[0][6]), round(x[0][7])))

display_flag = True
positives, negatives, percentage = test_images(images, attributes)
print("Correct: ", positives, "Incorrect: ", negatives, "Percentage: ", percentage)
