raw=[0.54, 0.5, 0.5, 0.98, 0.54, 0.68, 0.64, 0.6, 0.56, 0.56, 0.74, 0.86, 0.74, 0.78, 0.86] #112
band4raw=[0.54, 0.6, 0.58, 0.68, 0.6, 0.54, 0.56, 0.58, 0.6, 0.6, 0.6, 0.54, 0.52, 0.76, 0.7]#113
band2=[0.62, 1.0, 0.74, 0.74, 0.66, 0.54, 0.6, 0.66, 0.58, 0.64, 0.56, 0.56, 0.58, 0.72, 0.64]# 97
band22=[0.52, 1.0, 0.7, 0.58, 0.58, 0.58, 0.58, 0.58, 0.6, 0.66, 0.58, 0.56, 0.64, 0.82, 0.62]# 80
ensemble=[max(i,j) for i,j in zip(raw,band22)]
print(sum(ensemble)/len(ensemble))

# 97
# 107