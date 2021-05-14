from utils_A2D2 import *
import matplotlib.pyplot as plt

extrapolated = np.loadtxt("/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/A2D2/A2D2_Tutorial/all_images_extrapolated_distance")
detected = np.loadtxt("/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/A2D2/A2D2_Tutorial/all_images_detected_distance")
cars = np.loadtxt("images_upto5621_detected_cars04")
trucks = np.loadtxt("images_upto5621_detected_trucks01")
ped_bc = np.loadtxt("images_upto5621_detected_ped_bc01")

nbr_img = extrapolated.shape[0]
x = range(0, nbr_img)

print("Number of images where cars block the visibility: ", np.sum(cars))
print("Number of images where trucks block the visibility: ", np.sum(trucks))
print("Number of images where pedestrians or bicycles block the visibility: ", np.sum(ped_bc))

'''plt.scatter(x=x, y=extrapolated, alpha=0.25)
plt.xlabel("Picture")
plt.ylabel("Maximal extrapolated road distance")
plt.show()

plt.scatter(x=x, y=detected, alpha=0.25)
plt.xlabel("Picture")
plt.ylabel("Maximal detected road distance")
plt.show()

plt.hist(extrapolated, bins=400)
plt.xlim(0, 500)
plt.show()
plt.hist(detected, bins=100)
plt.xlim(0, 250)
plt.show()'''