import pandas as pd
import seaborn as sns
df = pd.read_excel('../Results.xlsx')

x = df.loc[(df.Classes == '3-classes') & (df.Approach == 'Model-centric')]
print(x.columns)
x = x["ID-INPUT WHEN RUN"].values
df = df.loc[(df.Approach == 'Model-centric')]
# df_3 = df.loc[(df.Classes == '3-classes') & (df.Approach == 'Model-centric')]
# df_5 = df.loc[(df.Classes == '5-classes') & (df.Approach == 'Model-centric')]
# df_8 = df.loc[(df.Classes == '8-classes') & (df.Approach == 'Model-centric')]
# print(df)
# print(df_3["F1-Score-Weighted"].values)


figure = sns.catplot(data=df, x="Model", y="F1-Score-Weighted", col="Classes", kind="bar", height=4, aspect=0.7) # instead of col hue?
figure.set_xticklabels(rotation=30, ha="right")
figure.set(ylim=(0.75, 1.0))
figure.savefig("Model-Centric_Improvements_all_classes_categorial.pdf")