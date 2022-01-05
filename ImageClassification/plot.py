import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel('../Results.xlsx')
x = df.loc[(df.Classes == '3-classes') & (df.Approach == 'Model-centric')]
print(x.columns)
x = x["ID-INPUT WHEN RUN"].values
df_3 = df.loc[(df.Classes == '3-classes') & (df.Approach == 'Model-centric')]
df_5 = df.loc[(df.Classes == '5-classes') & (df.Approach == 'Model-centric')]
df_8 = df.loc[(df.Classes == '8-classes') & (df.Approach == 'Model-centric')]
# print(df)
# print(df_3["F1-Score-Weighted"].values)

fig, ax = plt.subplots()



plt.bar(df_3["ID-INPUT WHEN RUN"].values, df_3["F1-Score-Weighted"].values, color='b', label='F1-Score-Weighted 3 Classes')
# for i, txt in enumerate(df_3["ID-INPUT WHEN RUN"].values):
#     ax.annotate(txt, (x[i] - 0.09, df_3["F1-Score-Weighted"].values[i] + 0.005))

# x = df_5["Model"].values
plt.bar(df_5["ID-INPUT WHEN RUN"].values, df_5["F1-Score-Weighted"].values, color='g', label='F1-Score-Weighted 5 Classes')
# for i, txt in enumerate(df_5["ID-INPUT WHEN RUN"].values):
#     ax.annotate(txt, (x[i] - 0.09, df_5["F1-Score-Weighted"].values[i] -0.009))

plt.bar(df_8["ID-INPUT WHEN RUN"].values, df_8["F1-Score-Weighted"].values, color='r', label='F1-Score-Weighted 8 Classes')
# for i, txt in enumerate(df_8["ID-INPUT WHEN RUN"].values):
#     ax.annotate(txt, (x[i] - 0.09, df_8["F1-Score-Weighted"].values[i] -0.009))

plt.xticks(rotation=-30)
plt.gcf().subplots_adjust(bottom=0.2)
plt.legend()
#plt.show()
plt.savefig("Model-Centric_Improvements_all_classes.pdf")