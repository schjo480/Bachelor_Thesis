from evaluation import *
from post_processing_utils import *

###################################################################################################################
######################Visualizations of the obtained results#######################################################
###################################################################################################################


color1 = '#005293'
color2 = '#E37222'
color3 = '#A2AD00'


a2d2 = pd.read_csv("/Results/A2D2/A2D2.csv")
#column names: timestamp, dist_until_blocked, max_detected, 99% detected, 95% detected, 90% detected, 85% detected,
# 80% detected, 75% detected, max_extrapolated, 99% extrapolated, 95% extrapolated, 90% extrapolated, 85% extrapolated,
# 80% extrapolated, 75% extrapolated, speed, acceleration, longitude, latitude, cars 50%, cars 40%, cars 30%,
# trucks 50%, trucks 40%, trucks 30%, pedestrian 50%, pedestrian 40%, pedestrian 30%, other 50%, other 40%, other 50%,
# time, hour, road_category, bad_weather, rush hour, type

nbr_frames = len(a2d2.index)

#Filter out non relevant frames here!
'''a2d2 = a2d2[(a2d2["99% extrapolated"] < 250) & (a2d2["99% extrapolated"] > 0)]
a2d2 = a2d2[(a2d2["max_extrapolated"] < 250) & (a2d2["max_extrapolated"] > 0)]
a2d2 = a2d2[(a2d2["dist_until_blocked"] > 0) & (a2d2["dist_until_blocked"] < 250)]
a2d2.to_csv("Results/A2D2/A2D2.csv", index=False)'''

#Slice the data
a2d2["bad_weather"] = a2d2["bad_weather"].astype(int)
a2d2.to_csv("Results/A2D2/A2D2.csv", index=False)
other = a2d2[a2d2["type"] == "other"]
rush_hour = a2d2[a2d2["rush_hour"] == 1]
not_rush = a2d2[a2d2["rush_hour"] == 0]
motorway = a2d2[a2d2["type"] == 'motorway']
primary = a2d2[(a2d2["type"] == 'primary') | (a2d2["type"] == 'trunk')]
secondary = a2d2[a2d2["type"] == 'secondary']
tertiary = a2d2[a2d2["type"] == 'tertiary']
unclassified = a2d2[a2d2["type"] == 'unclassified']
residential = a2d2[(a2d2["type"] == 'residential') | (a2d2["type"] == 'living_street')]
blocked = a2d2[(a2d2["cars 50%"] == 1) | (a2d2["cars 40%"] == 1) | (a2d2["trucks 50%"] == 1) | (a2d2["trucks 40%"] == 1)
               | (a2d2["pedestrian 50%"] == 1) | (a2d2["pedestrian 40%"] == 1)]
not_blocked = a2d2[((a2d2["cars 50%"] == 0) & (a2d2["cars 40%"] == 0) | (a2d2["cars 30%"] == 1)) &
                   ((a2d2["trucks 50%"] == 0) & (a2d2["trucks 40%"] == 0) | (a2d2["trucks 30%"] == 1)) &
                   ((a2d2["pedestrian 50%"] == 0) & (a2d2["pedestrian 40%"] == 0) | (a2d2["pedestrian 30%"] == 1))]
not_bad = a2d2[a2d2["bad_weather"] == 0]
bad = a2d2[a2d2["bad_weather"] == 1]

#Plots for A2D2

plt.hist(motorway["99% extrapolated"], color=color1, bins=150, alpha=0.7, label="Motorway", weights=
np.ones(len(motorway["99% extrapolated"])) / len(motorway["99% extrapolated"]))
plt.hist(other["99% extrapolated"], color=color2, bins=100, alpha=0.7, label="Other roads", weights=
np.ones(len(other["99% extrapolated"])) / len(other["99% extrapolated"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.title("A2D2", size=36)
plt.legend(loc='upper right', fontsize=25)
plt.xlim((0, 130))
plt.ylim(0, 0.15)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid(b=True, which='major', axis='y')
plt.axvline(np.median(motorway["99% extrapolated"]), color=color1, linestyle='dashed')
plt.axvline(np.median(other["99% extrapolated"]), color=color2, linestyle='dashed')
plt.xlabel("Calculated distance in m", size=30)
plt.ylabel("Percentage of frames", size=30)
plt.figtext(x=0.61, y=0.54, s="Median:\nMotorway = 43.79m\nOther = 40.56m", size=25)
plt.show()


plt.hist(motorway["dist_until_blocked"], color=color1, bins=90, alpha=0.7, label="Motorway", weights=
np.ones(len(motorway["dist_until_blocked"])) / len(motorway["dist_until_blocked"]))
plt.hist(primary["dist_until_blocked"], color=color2, bins=90, alpha=0.7, label="Primary roads", weights=
np.ones(len(primary["dist_until_blocked"])) / len(primary["dist_until_blocked"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.title("A2D2", size=36)
plt.legend(loc='upper right', fontsize=25)
plt.xlim((0, 130))
plt.ylim(0, 0.15)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid(b=True, which='major', axis='y')
plt.axvline(np.median(motorway["dist_until_blocked"]), color=color1, linestyle='dashed')
plt.axvline(np.median(primary["dist_until_blocked"]), color=color2, linestyle='dashed')
plt.xlabel("Calculated distance in m", size=30)
plt.ylabel("Percentage of frames", size=30)
plt.figtext(x=0.61, y=0.54, s="Median:\nMotorway = 43.89 m\nPrimary = 37.23 m", size=25)
plt.show()

plt.hist(not_blocked["99% extrapolated"], color=color1, bins=100, alpha=0.7, label="Not blocked", weights=
np.ones(len(not_blocked["99% extrapolated"])) / len(not_blocked["99% extrapolated"]))
plt.hist(blocked["dist_until_blocked"], color=color2, bins=70, alpha=0.7, label="Blocked", weights=
np.ones(len(blocked["dist_until_blocked"])) / len(blocked["dist_until_blocked"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.legend(loc='upper right', fontsize=25)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.axvline(np.median(not_blocked["99% extrapolated"]), color=color1, linestyle='dashed')
plt.axvline(np.median(blocked["dist_until_blocked"]), color=color2, linestyle='dashed')
plt.xlim((0, 130))
plt.ylim(0, 0.15)
plt.grid(b=True, which='major', axis='y')
plt.figtext(x=0.6, y=0.54, s="Median:\nNot blocked = 42.10 m\nBlocked = 12.42 m", size=25)
plt.title("A2D2", size=36)
plt.xlabel("Calculated distance in m", size=30)
plt.ylabel("Percentage of frames", size=30)
plt.show()


plt.hist(not_blocked["99% detected"], color=color1, bins=160, alpha=0.7, label="Not blocked", weights=
np.ones(len(not_blocked["99% detected"])) / len(not_blocked["99% detected"]))
plt.hist(blocked["99% detected"], color=color2, bins=70, alpha=0.7, label="Blocked", weights=
np.ones(len(blocked["99% detected"])) / len(blocked["99% detected"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.legend(loc='upper right', fontsize=25)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.axvline(np.median(not_blocked["99% detected"]), color=color1, linestyle='dashed')
plt.axvline(np.median(blocked["99% detected"]), color=color2, linestyle='dashed')
plt.xlim((0, 130))
plt.ylim(0, 0.15)
plt.grid(b=True, which='major', axis='y')
plt.figtext(x=0.6, y=0.54, s="Median:\nNot blocked = 28.45 m\nBlocked = 30.45 m", size=25)
plt.title("A2D2", size=36)
plt.xlabel("Detected distance in m", size=30)
plt.ylabel("Percentage of frames", size=30)
plt.show()


plt.hist(a2d2["99% detected"], bins=150, color=color1, label=["Median", "Mean"], weights=
np.ones(len(a2d2["99% detected"])) / len(a2d2["99% detected"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.axvline(np.median(a2d2["99% detected"]), color="black", linestyle='dashed')
plt.axvline(np.mean(a2d2["99% detected"]), color="red", linestyle='dashed')
plt.xlim(0, 130)
plt.ylim(0, 0.15)
plt.grid(b=True, which='major', axis='y')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.title("A2D2", size=36)
plt.figtext(x=0.55, y=0.63, s="Median = 28.75 m\nMean = 29.84 m", size=25)
plt.xlabel("Detected distance in m", size=30)
plt.ylabel("Number of frames", size=30)
plt.show()


names = ["Clear weather", "Bad weather (rain, fog)"]

sns.boxplot(data=[not_bad["99% extrapolated"], bad["99% extrapolated"]], palette=[color1, color2])
plt.legend(["Clear", "Bad"], loc="upper right", fontsize=25)
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color(color1)
leg.legendHandles[1].set_color(color2)
plt.ylim(0, 250)
plt.xticks(None)
plt.yticks(fontsize=30)
plt.title("A2D2", size=36)
ax.set_xticklabels(names, size=30)
plt.ylabel("Calculated distance in m", size=30)
plt.figtext(x=0.42, y=0.7, s="Standard deviations:\nClear = 9.99 m\nBad = 28.77 m", size=25)
plt.figtext(x=0.42, y=0.5, s="Median:\nClear = 42.31 m\nBad = 45.24 m", size=25)
plt.show()

'''plt.scatter(x_not_bad, not_bad["99% extrapolated"], alpha=0.4, color=color1)
plt.scatter(x_bad, bad["99% extrapolated"], alpha=0.4, color=color2)
plt.legend(["Clear weather", "Bad weather (fog, rain)"], loc="upper center")
plt.title("Calculated distance for every frame (99th percentile)", size=11)
plt.xlabel("Frame", size=11)
plt.ylabel("Calculated distance in m", size=11)
plt.grid(b=True, which='major', axis='y')
plt.savefig("Results/A2D2/Plots/Scatter Influence_of_Bad_weather calculated (99th percentile)).pdf", dpi=1000)
plt.show()'''

print(a2d2.type.value_counts())
road_types = {'Motorway': 2847/nbr_frames,'Secondary': 1904/nbr_frames,'Primary': 990/nbr_frames,'Other': \
    712/nbr_frames,'Tertiary': 398/nbr_frames,'Trunk': 166/nbr_frames,'Residential': 108/nbr_frames}
names = list(road_types.keys())
val = list(road_types.values())

plt.bar(x=names, height=val, color=color1)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax = plt.gca()
plt.grid(b=True, which='major', axis='y')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Percentage of frames", size=30)
plt.ylim(0, 0.4)
ax.set_xticklabels(names, rotation = 45, ha="right")
plt.title("A2D2", size=36)
plt.show()

block = {'Cars low': 1824/nbr_frames, 'Cars medium': 839/nbr_frames, 'Cars high': 522/nbr_frames, 'Trucks low': \
    602/nbr_frames, 'Trucks medium': 150/nbr_frames, 'Trucks high': 94/nbr_frames, 'Pedestrians low': 147/nbr_frames,\
         'Pedestrians medium': 50/nbr_frames, 'Pedestrians high': 86/nbr_frames}
names = list(block.keys())
val = list(block.values())

plt.bar(x=names, height=val, color=color1)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax = plt.gca()
plt.grid(b=True, which='major', axis='y')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Percentage of frames", size=30)
plt.ylim(0, 0.55)
ax.set_xticklabels(names, rotation=45, ha="right")
plt.title("A2D2", size=36)
plt.show()


'''kitti09_26 = pd.read_csv("/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/Final/Results/KITTI/KITTI_09_26.csv")
kitti09_28 = pd.read_csv("/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/Final/Results/KITTI/KITTI_09_28.csv")
kitti09_29 = pd.read_csv("/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/Final/Results/KITTI/KITTI_09_29.csv")
kitti09_30 = pd.read_csv("/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/Final/Results/KITTI/KITTI_09_30.csv")
kitti10_03 = pd.read_csv("/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/Final/Results/KITTI/KITTI_10_03.csv")'''
kitti = pd.read_csv("/Results/KITTI/KITTI.csv")
#column names: timestamp, max_detected, 99% detected, 95% detected, 90% detected, 85% detected, 80% detected, \
# 75% detected, speed, acceleration, longitude, latitude, hour, road category, rush hour, type

#Filter out non relevant frames here!
'''kitti = kitti[kitti["99% extrapolated"] > 0]
kitti = kitti[kitti["99% extrapolated"] < 250]
kitti = kitti[kitti["road category"] == "highway"]'''

nbr_frames = 4182.0

motorway = kitti[kitti["type"] == 'motorway']
primary = kitti[(kitti["type"] == 'primary') | (kitti["type"] == 'trunk')]
secondary = kitti[kitti["type"] == 'secondary']
tertiary = kitti[kitti["type"] == 'tertiary']
unclassified = kitti[kitti["type"] == 'unclassified']
residential = kitti[(kitti["type"] == 'residential') | (kitti["type"] == 'living_street')]

#Plots for KITTI

plt.hist(motorway["99% extrapolated"], color=color1, bins=60, alpha=0.7, label="Motorway", weights=
np.ones(len(motorway["99% extrapolated"])) / len(motorway["99% extrapolated"]))
plt.hist(secondary["99% extrapolated"], color=color2, bins=150, alpha=0.7, label="Secondary roads", weights=
np.ones(len(secondary["99% extrapolated"])) / len(secondary["99% extrapolated"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.title("KITTI", size=36)
plt.legend(loc='upper right', fontsize=25)
plt.xlim((0, 130))
plt.ylim(0, 0.15)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid(b=True, which='major', axis='y')
plt.axvline(np.median(motorway["99% extrapolated"]), color=color1, linestyle='dashed')
plt.axvline(np.median(secondary["99% extrapolated"]), color=color2, linestyle='dashed')
plt.xlabel("Calculated distance in m", size=30)
plt.ylabel("Percentage of frames", size=30)
plt.figtext(x=0.61, y=0.54, s="Median:\nMotorway = 47.41 m\nSecondary = 41.39 m", size=25)
plt.show()

plt.hist(kitti["95% extrapolated"], bins=90, color=color1, label=["Median", "Mean"], weights=
np.ones(len(kitti["95% extrapolated"])) / len(kitti["95% extrapolated"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.axvline(np.median(kitti["95% extrapolated"]), color="black", linestyle='dashed')
plt.axvline(np.mean(kitti["95% extrapolated"]), color="red", linestyle='dashed')
plt.xlim(0, 130)
plt.ylim(0, 0.15)
plt.grid(b=True, which='major', axis='y')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title("KITTI", size=36)
plt.figtext(x=0.55, y=0.63, s="Median = 27.16m\nMean = 29.75m", size=25)
plt.xlabel("Calculated distance in m", size=30)
plt.ylabel("Percentage of frames", size=30)
plt.show()

road_types = {'Primary': 1458/nbr_frames,'Trunk': 833/nbr_frames,'Secondary': 803/nbr_frames,'Other': 418/nbr_frames,
              'Tertiary': 307/nbr_frames,'Residential': 253/nbr_frames,'Motorway': 107/nbr_frames}
names = list(road_types.keys())
val = list(road_types.values())


plt.bar(x=names, height=val, color=color1)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax = plt.gca()
plt.grid(b=True, which='major', axis='y')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Percentage of frames", size=30)
ax.set_xticklabels(names, rotation = 45, ha="right")
plt.ylim(0, 0.4)
plt.title("KITTI", size=36)
plt.show()



apollo = pd.read_csv("/Results/ApolloScape Stereo/Apolloscape_stereo.csv")

#Filter out non relevant frames here!
apollo["max_detected"] = apollo["max_detected"].replace('inf', np.percentile(apollo["max_detected"], 99))

#Plots for Apolloscape_stereo

plt.hist(apollo["max_detected"], bins=110, color=color1, label=["Median", "Mean"], weights=
np.ones(len(apollo["max_detected"])) / len(apollo["max_detected"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.axvline(np.median(apollo["max_detected"]), color="black", linestyle='dashed')
plt.axvline(np.mean(apollo["max_detected"]), color="red", linestyle='dashed')
plt.xlim(0, 130)
plt.ylim(0, 0.15)
plt.grid(b=True, which='major', axis='y')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title("Apolloscape Stereo", size=36)
plt.figtext(x=0.55, y=0.63, s="Median = 25.89m\nMean = 25.72m", size=25)
plt.xlabel("Detected distance in m", size=30)
plt.ylabel("Number of frames", size=30)
plt.show()


sem_apollo = pd.read_csv("/Results/Apolloscape Semantic/Apolloscape_semantic new.csv")

nbr_frames = len(sem_apollo.index)

#Plots for ApolloScape Semantic

not_blocked = sem_apollo[((sem_apollo["cars 50%"] == 0) & (sem_apollo["cars 40%"] == 0) | (sem_apollo["cars 30%"] == 1))
                         & ((sem_apollo["trucks 50%"] == 0) & (sem_apollo["trucks 40%"] == 0) |
                            (sem_apollo["trucks 30%"] == 1)) & ((sem_apollo["pedestrian 50%"] == 0) &
                                                                (sem_apollo["pedestrian 40%"] == 0) |
                                                                (sem_apollo["pedestrian 30%"] == 1))]
blocked = sem_apollo[(sem_apollo["cars 50%"] == 1) | (sem_apollo["cars 40%"] == 1) | (sem_apollo["trucks 50%"] == 1) |
                     (sem_apollo["trucks 40%"] == 1) | (sem_apollo["pedestrian 50%"] == 1) |
                     (sem_apollo["pedestrian 40%"] == 1)]

plt.hist(not_blocked["99% detected"], color=color1, bins=60, alpha=0.7, label="Not blocked", weights=
np.ones(len(not_blocked["99% detected"])) / len(not_blocked["99% detected"]))
plt.hist(blocked["dist_until_blocked"], color=color2, bins=180, alpha=0.7, label="Blocked", weights=
np.ones(len(blocked["dist_until_blocked"])) / len(blocked["dist_until_blocked"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.legend(loc='upper right', fontsize=25)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.axvline(np.median(not_blocked["99% detected"]), color=color1, linestyle='dashed')
plt.axvline(np.median(blocked["dist_until_blocked"]), color=color2, linestyle='dashed')
plt.xlim((0, 130))
plt.ylim(0, 0.15)
plt.grid(b=True, which='major', axis='y')
plt.figtext(x=0.6, y=0.54, s="Median:\nNot blocked = 61.20 m\nBlocked = 12.11 m", size=25)
plt.title("Apolloscape Scene Parsing", size=36)
plt.xlabel("Detected distance in m", size=30)
plt.ylabel("Percentage of frames", size=30)
plt.show()

block = {'Cars low': 1087/nbr_frames, 'Cars medium': 189/nbr_frames, 'Cars high': 85/nbr_frames, 'Trucks low': \
    185/nbr_frames, 'Trucks medium': 29/nbr_frames, 'Trucks high': 16/nbr_frames, 'Pedestrians low': 19/nbr_frames,\
         'Pedestrians medium': 2/nbr_frames, 'Pedestrians high': 1/nbr_frames}
names = list(block.keys())
val = list(block.values())

plt.bar(x=names, height=val, color=color1)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax = plt.gca()
plt.grid(b=True, which='major', axis='y')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Percentage of frames", size=30)
plt.ylim(0, 0.55)
ax.set_xticklabels(names, rotation=45, ha="right")
plt.title("ApolloScape Scene Parsing", size=36)
plt.show()

plt.hist(sem_apollo["99% detected"], bins=40, color=color1, label=["Median", "Mean"], weights=
np.ones(len(sem_apollo["99% detected"])) / len(sem_apollo["99% detected"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.axvline(np.median(sem_apollo["99% detected"]), color="black", linestyle='dashed')
plt.axvline(np.mean(sem_apollo["99% detected"]), color="red", linestyle='dashed')
plt.xlim(0, 130)
plt.ylim(0, 0.15)
plt.grid(b=True, which='major', axis='y')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title("Apolloscape Semantic", size=36)
plt.figtext(x=0.55, y=0.63, s="Median = 61.20m\nMean = 61.67m", size=25)
plt.xlabel("Detected distance in m", size=30)
plt.ylabel("Number of frames", size=30)
plt.show()



oxford = pd.read_csv("/Results/Oxford/Oxford.csv")
#column names: timestamp, max_detected, 99% detected, 95% detected, 90% detected, 85% detected, 80% detected, \
# 75% detected, speed, longitude, latitude, time, hour, road category, bad_weather, rush hour, traffic, type, night


#Filter out non relevant frames here!
'''oxford = oxford[oxford["99% detected"] < 250]'''
not_bad = oxford[oxford["bad weather"] == 0]
bad = oxford[oxford["bad weather"] == 1]

day = oxford[oxford["night"] == 0]
night = oxford[oxford["night"] == 1]

no_rush = oxford[oxford["rush_hour"] == 0]
rush = oxford[oxford["rush_hour"] == 1]


#Plots for Oxford

plt.hist(oxford["90% detected"], bins=150, color=color1, label=["Median", "Mean"], weights=
np.ones(len(oxford["90% detected"])) / len(oxford["90% detected"]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.axvline(np.median(oxford["90% detected"]), color="black", linestyle='dashed')
plt.axvline(np.mean(oxford["90% detected"]), color="red", linestyle='dashed')
plt.xlim(0, 130)
plt.ylim(0, 0.15)
plt.grid(b=True, which='major', axis='y')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title("Oxford", size=36)
plt.figtext(x=0.55, y=0.63, s="Median = 33.58m\nMean = 36.31m", size=25)
plt.xlabel("Detected distance in m", size=30)
plt.ylabel("Percentage of frames", size=30)
plt.show()


names = ["Clear weather", "Bad weather (rain, fog)"]

sns.boxplot(data=[not_bad["99% detected"], bad["99% detected"]], palette=[color1, color2])
plt.legend(["Clear", "Bad"], loc="upper right", fontsize=25)
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color(color1)
leg.legendHandles[1].set_color(color2)
plt.ylim(0, 250)
plt.xticks(None)
plt.yticks(fontsize=25)
plt.title("Oxford", size=36)
ax.set_xticklabels(names, size=30)
plt.ylabel("Detected distance in m", size=30)
plt.savefig("../../Ausarbeitung/Bilder/4/Oxford/Boxplot Clear vs Bad 99%.pdf")
plt.figtext(x=0.42, y=0.7, s="Standard deviations:\nClear = 27.75m\nBad = 33.46m", size=25)
plt.figtext(x=0.42, y=0.5, s="Median:\nClear = 45.58m\nBad = 42.29m", size=25)
plt.savefig("Results/Oxford/Plots/Boxplot Clear vs Bad 99%.pdf")
plt.show()
