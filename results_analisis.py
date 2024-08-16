import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


base_path=r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\data"
df=pd.read_csv(r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\results1.csv")
#%%
print("Average IoU:",df["IoU"].mean())
print("Average recall:",df["recall"].mean())
print("Average presicion:",df["precisions"].mean())

plt.hist(df["IoU"])
plt.title("IoUs")
plt.show()
plt.hist(df["recall"])
plt.title("recalls")
plt.show()
plt.hist(df["precisions"])
plt.title("precisions")

plt.show()
#%%

# plt.hist(df["IoU"])
# plt.title("IoUs")
# plt.show()
# plt.hist(df["recall"])
# plt.title("recalls")
# plt.show()
# plt.hist(df["precisions"])
# plt.title("precisions")
#
# plt.show()

#%%
df["second_phase"]=np.sum(np.array(df[['top_left_second_phase', 'bottom_left_second_phase',
       'bottom_right_second_phase', 'top_right_second_phase']]),1)
df["corners_found"]=np.sum(np.array(df[['top_left_second_phase_inside', 'bottom_left_second_phase_inside',
       'bottom_right_second_phase_inside', 'top_right_second_phase_inside']]),1)
#%%
plt.hist(df["second_phase"])
plt.title("second_phase")
plt.show()
plt.hist(df["corners_found"])
plt.title("corners_found")
plt.show()
#%%
for second_phase_numbers in range(5):

       slice_df=df[df["second_phase"]==second_phase_numbers]
       print(f"Si se pasan {second_phase_numbers} a la segunda fase")
       print("       Average IoU:", slice_df["IoU"].mean())
       print("       Average recall:", slice_df["recall"].mean())
       print("       Average presicion:", slice_df["precisions"].mean())

       plt.hist(slice_df["IoU"])
       title_name=f"IoUs given f{second_phase_numbers} corners pass the first phase"
       plt.title(title_name)
       plt.savefig(os.path.join(base_path,title_name.replace(" ","_")+".png"))
       plt.hist(slice_df["recall"])
       title_name=f"recalls given f{second_phase_numbers} corners pass the first phase"
       plt.title(title_name)
       plt.savefig(os.path.join(base_path,title_name.replace(" ","_")+".png"))
       plt.hist(slice_df["precisions"])
       title_name=f"precisions given f{second_phase_numbers} corners pass the first phase"
       plt.title(title_name)
       plt.savefig(os.path.join(base_path,title_name.replace(" ","_")+".png"))

#%%
for corners_found in range(5):

       slice_df=df[df["second_phase"]==corners_found]
       print(f"Si se encuentran {corners_found} esquinas en la  a la segunda fase")
       print("       Average IoU:", slice_df["IoU"].mean())
       print("       Average recall:", slice_df["recall"].mean())
       print("       Average presicion:", slice_df["precisions"].mean())

       plt.hist(slice_df["IoU"])
       tittle=f"IoUs given f{corners_found} corner inside second phase"
       plt.title(tittle)
       plt.savefig(os.path.join(base_path,tittle.replace(" ","_")+".png"))
       plt.hist(slice_df["recall"])
       tittle=f"recalls given f{corners_found} corner inside second phase"
       plt.title(tittle)
       plt.savefig(os.path.join(base_path,tittle.replace(" ","_")+".png"))
       plt.hist(slice_df["precisions"])
       tittle=f"precisions given f{corners_found} corner inside second phase"
       plt.title(tittle)
       plt.savefig(os.path.join(base_path,tittle.replace(" ","_")+".png"))
       # plt.show()