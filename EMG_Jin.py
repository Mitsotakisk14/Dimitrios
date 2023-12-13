import itertools
import statistics
import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Opening the document and cutting the data
with open('jin set 1.txt') as jin1:
  for line in itertools.islice(jin1, 2842, None):
    data_jin1 = pd.read_csv(jin1, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_j1 = data_jin1[0: 770]

with open('jin set 2.txt') as jin2:
  for line in itertools.islice(jin2, 12692, None):
    data_jin2 = pd.read_csv(jin2, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_j2 = data_jin2[0: 530]

with open('jitse set 1 new.txt') as jitse1:
  for line in itertools.islice(jitse1, 2102, None):
    data_jitse1 = pd.read_csv(jitse1, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_jit1 = data_jitse1[0: 720]

with open('jitse set 2 new.txt') as jitse2:
  for line in itertools.islice(jitse2, 4132, None):
    data_jitse2 = pd.read_csv(jitse2, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_jit2 = data_jitse2[0: 630]

with open('kevin_set_1.txt') as kev1:
  for line in itertools.islice(kev1, 4462, None):
    data_kev1 = pd.read_csv(kev1, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_k1 = data_kev1[0: 940]

with open('KEVIN_FINAL_SET_2.txt') as kev2:
  for line in itertools.islice(kev2, 7192, None):
    data_kev2 = pd.read_csv(kev2, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_k2 = data_kev2[0: 700]

with open('marco set 1.txt') as marco1:
  for line in itertools.islice(marco1, 2622, None):
    data_marco1 = pd.read_csv(marco1, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_m1 = data_marco1[0: 1200]

with open('marco set 2.txt') as marco2:
  for line in itertools.islice(marco2, 4122, None):
    data_marco2 = pd.read_csv(marco2, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_m2 = data_marco2[0: 820]

with open('dim set 1.txt') as dim1:
  for line in itertools.islice(dim1, 4802, None):
    data_dim1 = pd.read_csv(dim1, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_d1 = data_dim1[0: 960]

with open('dim set 2.txt') as dim2:
  for line in itertools.islice(dim2, 7552, None):
    data_dim2 = pd.read_csv(dim2, sep="\t", names=["col1", "col2", "col3", "col4", "col5","col6","col7","col8","col9"])
    data_d2 = data_dim2[0: 850]

# time(in seconds)
col1_j1 = data_j1["col1"]/100
col1_j2 = data_j2["col1"]/100

col1_jit1 = data_jit1["col1"]/100
col1_jit2 = data_jit2["col1"]/100

col1_k1 = data_k1["col1"]/100
col1_k2 = data_k2["col1"]/100

col1_m1 = data_m1["col1"]/100
col1_m2 = data_m2["col1"]/100

col1_d1 = data_d1["col1"]/100
col1_d2 = data_d2["col1"]/100

# R shoulder
col3_j1 = data_j1["col3"]
col3_j2 = data_j2["col3"]

col3_jit1 = data_jit1["col3"]
col3_jit2 = data_jit2["col3"]

col3_k1 = data_k1["col3"]
col3_k2 = data_k2["col3"]

col3_m1 = data_m1["col3"]
col3_m2 = data_m2["col3"]

col3_d1 = data_d1["col3"]
col3_d2 = data_d2["col3"]

# L shoulder
col4_j1 = data_j1["col4"]
col4_j2 = data_j2["col4"]

col4_jit1 = data_jit1["col4"]
col4_jit2 = data_jit2["col4"]

col4_k1 = data_k1["col4"]
col4_k2 = data_k2["col4"]

col4_m1 = data_m1["col4"]
col4_m2 = data_m2["col4"]

col4_d1 = data_d1["col4"]
col4_d2 = data_d2["col4"]

# L chest
col5_j1 = data_j1["col5"]
col5_j2 = data_j2["col5"]

col5_jit1 = data_jit1["col5"]
col5_jit2 = data_jit2["col5"]

col5_k1 = data_k1["col5"]
col5_k2 = data_k2["col5"]

col5_m1 = data_m1["col5"]
col5_m2 = data_m2["col5"]

col5_d1 = data_d1["col5"]
col5_d2 = data_d2["col5"]

# R chest
col6_j1 = data_j1["col6"]
col6_j2 = data_j2["col6"]

col6_jit1 = data_jit1["col6"]
col6_jit2 = data_jit2["col6"]

col6_k1 = data_k1["col6"]
col6_k2 = data_k2["col6"]

col6_m1 = data_m1["col6"]
col6_m2 = data_m2["col6"]

col6_d1 = data_d1["col6"]
col6_d2 = data_d2["col6"]

# L triceps
col7_j1 = data_j1["col7"]
col7_j2 = data_j2["col7"]

col7_jit1 = data_jit1["col7"]
col7_jit2 = data_jit2["col7"]

col7_k1 = data_k1["col7"]
col7_k2 = data_k2["col7"]

col7_m1 = data_m1["col7"]
col7_m2 = data_m2["col7"]

col7_d1 = data_d1["col7"]
col7_d2 = data_d2["col7"]

# R triceps
col8_j1 = data_j1["col8"]
col8_j2 = data_j2["col8"]

col8_jit1 = data_jit1["col8"]
col8_jit2 = data_jit2["col8"]

col8_k1 = data_k1["col8"]
col8_k2 = data_k2["col8"]

col8_m1 = data_m1["col8"]
col8_m2 = data_m2["col8"]

col8_d1 = data_d1["col8"]
col8_d2 = data_d2["col8"]

# Do the sum of all the muscle and put them in the filter
sum_j1 = col3_j1 + col4_j1 + col5_j1 + col6_j1 + col7_j1 + col8_j1
sum_j2 = col3_j2 + col4_j2 + col5_j2 + col6_j2 + col7_j2 + col8_j2
sum_jit1 = col3_jit1 + col4_jit1 + col5_jit1 + col6_jit1 + col7_jit1 + col8_jit1
sum_jit2 = col3_jit2 + col4_jit2 + col5_jit2 + col6_jit2 + col7_jit2 + col8_jit2
sum_k1 = col3_k1 + col4_k1 + col5_k1 + col6_k1 + col7_k1 + col8_k1
sum_k2 = col3_k2 + col4_k2 + col5_k2 + col6_k2 + col7_k2 + col8_k2
sum_m1 = col3_m1 + col4_m1 + col5_m1 + col6_m1 + col7_m1 + col8_m1
sum_m2 = col3_m2 + col4_m2 + col5_m2 + col6_m2 + col7_m2 + col8_m2
sum_d1 = col3_d1 + col4_d1 + col5_d1 + col6_d1 + col7_d1 + col8_d1
sum_d2 = col3_d2 + col4_d2 + col5_d2 + col6_d2 + col7_d2 + col8_d2

def filter(N, Wn, input_data, window_size_smooth, window_size_rms):
  # normalization
  input_data_zero = input_data - input_data.mean()
  input_data_norm = (input_data_zero - input_data_zero.min()) / (input_data_zero.max() - input_data_zero.min())

  # band-pass filter
  b,a = signal.butter(N, Wn, "bandpass", False, "ba", 100)
  bandpass_result = signal.filtfilt(b, a, input_data_norm)

  # rectification filter
  rectification_result = abs(bandpass_result)

  # smoothing filter
  smooth_result = np.convolve(rectification_result, np.ones(window_size_smooth)/window_size_smooth, mode='same')

  # RMS
  half_window_rms = window_size_rms // 2

  # Pad the line to handle edge cases
  padded_line = np.pad(smooth_result, (half_window_rms, half_window_rms), mode='reflect')

  # Calculate the RMS for each point using a sliding window
  rms_result = np.sqrt(np.convolve(padded_line ** 2, np.ones(window_size_rms) / window_size_rms, mode='valid'))

  return rms_result

# normal set
rms_col3_j1 = filter(5, [2,45], col3_j1, 20, 15)
rms_col4_j1 = filter(5, [2,45], col4_j1, 20, 15)
rms_col5_j1 = filter(5, [2,45], col5_j1, 20, 15)
rms_col6_j1 = filter(5, [2,45], col6_j1, 20, 15)
rms_col7_j1 = filter(5, [2,45], col7_j1, 20, 15)
rms_col8_j1 = filter(5, [2,45], col8_j1, 20, 15)

rms_col3_jit1 = filter(5, [2,45], col3_jit1, 20, 15)
rms_col4_jit1 = filter(5, [2,45], col4_jit1, 20, 15)
rms_col5_jit1 = filter(5, [2,45], col5_jit1, 20, 15)
rms_col6_jit1 = filter(5, [2,45], col6_jit1, 20, 15)
rms_col7_jit1 = filter(5, [2,45], col7_jit1, 20, 15)
rms_col8_jit1 = filter(5, [2,45], col8_jit1, 20, 15)

rms_col3_k1 = filter(5, [2,45], col3_k1, 20, 15)
rms_col4_k1 = filter(5, [2,45], col4_k1, 20, 15)
rms_col5_k1 = filter(5, [2,45], col5_k1, 20, 15)
rms_col6_k1 = filter(5, [2,45], col6_k1, 20, 15)
rms_col7_k1 = filter(5, [2,45], col7_k1, 20, 15)
rms_col8_k1 = filter(5, [2,45], col8_k1, 20, 15)

rms_col3_m1 = filter(5, [2,45], col3_m1, 20, 15)
rms_col4_m1 = filter(5, [2,45], col4_m1, 20, 15)
rms_col5_m1 = filter(5, [2,45], col5_m1, 20, 15)
rms_col6_m1 = filter(5, [2,45], col6_m1, 20, 15)
rms_col7_m1 = filter(5, [2,45], col7_m1, 20, 15)
rms_col8_m1 = filter(5, [2,45], col8_m1, 20, 15)

rms_col3_d1 = filter(5, [2,45], col3_d1, 20, 15)
rms_col4_d1 = filter(5, [2,45], col4_d1, 20, 15)
rms_col5_d1 = filter(5, [2,45], col5_d1, 20, 15)
rms_col6_d1 = filter(5, [2,45], col6_d1, 20, 15)
rms_col7_d1 = filter(5, [2,45], col7_d1, 20, 15)
rms_col8_d1 = filter(5, [2,45], col8_d1, 20, 15)

# test set
rms_col3_j2 = filter(5, [2,45], col3_j2, 20, 15)
rms_col4_j2 = filter(5, [2,45], col4_j2, 20, 15)
rms_col5_j2 = filter(5, [2,45], col5_j2, 20, 15)
rms_col6_j2 = filter(5, [2,45], col6_j2, 20, 15)
rms_col7_j2 = filter(5, [2,45], col7_j2, 20, 15)
rms_col8_j2 = filter(5, [2,45], col8_j2, 20, 15)

rms_col3_jit2 = filter(5, [2,45], col3_jit2, 20, 15)
rms_col4_jit2 = filter(5, [2,45], col4_jit2, 20, 15)
rms_col5_jit2 = filter(5, [2,45], col5_jit2, 20, 15)
rms_col6_jit2 = filter(5, [2,45], col6_jit2, 20, 15)
rms_col7_jit2 = filter(5, [2,45], col7_jit2, 20, 15)
rms_col8_jit2 = filter(5, [2,45], col8_jit2, 20, 15)

rms_col3_k2 = filter(5, [2,45], col3_k2, 20, 15)
rms_col4_k2 = filter(5, [2,45], col4_k2, 20, 15)
rms_col5_k2 = filter(5, [2,45], col5_k2, 20, 15)
rms_col6_k2 = filter(5, [2,45], col6_k2, 20, 15)
rms_col7_k2 = filter(5, [2,45], col7_k2, 20, 15)
rms_col8_k2 = filter(5, [2,45], col8_k2, 20, 15)

rms_col3_m2 = filter(5, [2,45], col3_m2, 20, 15)
rms_col4_m2 = filter(5, [2,45], col4_m2, 20, 15)
rms_col5_m2 = filter(5, [2,45], col5_m2, 20, 15)
rms_col6_m2 = filter(5, [2,45], col6_m2, 20, 15)
rms_col7_m2 = filter(5, [2,45], col7_m2, 20, 15)
rms_col8_m2 = filter(5, [2,45], col8_m2, 20, 15)

rms_col3_d2 = filter(5, [2,45], col3_d2, 20, 15)
rms_col4_d2 = filter(5, [2,45], col4_d2, 20, 15)
rms_col5_d2 = filter(5, [2,45], col5_d2, 20, 15)
rms_col6_d2 = filter(5, [2,45], col6_d2, 20, 15)
rms_col7_d2 = filter(5, [2,45], col7_d2, 20, 15)
rms_col8_d2 = filter(5, [2,45], col8_d2, 20, 15)

rms_j1 = filter(5, [2,45], sum_j1, 20, 15)
rms_j2 = filter(5, [2,45], sum_j2, 20, 15)
rms_jit1 = filter(5, [2,45], sum_jit1, 20, 15)
rms_jit2 = filter(5, [2,45], sum_jit2, 20, 15)
rms_k1 = filter(5, [2,45], sum_k1, 20, 15)
rms_k2 = filter(5, [2,45], sum_k2, 20, 15)
rms_m1 = filter(5, [2,45], sum_m1, 20, 15)
rms_m2 = filter(5, [2,45], sum_m2, 20, 15)
rms_d1 = filter(5, [2,45], sum_d1, 20, 15)
rms_d2 = filter(5, [2,45], sum_d2, 20, 15)
# create a list for x-axis
def list_create(length):
  list = []
  for i in range(0, length):
    list.append(100 * i/length)
    i = i + 1
  return list

# mean value
mean_col3_j1 = statistics.mean(rms_col3_j1)
mean_col3_j2 = statistics.mean(rms_col3_j2)
mean_col4_j1 = statistics.mean(rms_col4_j1)
mean_col4_j2 = statistics.mean(rms_col4_j2)
mean_col5_j1 = statistics.mean(rms_col5_j1)
mean_col5_j2 = statistics.mean(rms_col5_j2)
mean_col6_j1 = statistics.mean(rms_col6_j1)
mean_col6_j2 = statistics.mean(rms_col6_j2)
mean_col7_j1 = statistics.mean(rms_col7_j1)
mean_col7_j2 = statistics.mean(rms_col7_j2)
mean_col8_j1 = statistics.mean(rms_col8_j1)
mean_col8_j2 = statistics.mean(rms_col8_j2)

mean_col3_jit1 = statistics.mean(rms_col3_jit1)
mean_col3_jit2 = statistics.mean(rms_col3_jit2)
mean_col4_jit1 = statistics.mean(rms_col4_jit1)
mean_col4_jit2 = statistics.mean(rms_col4_jit2)
mean_col5_jit1 = statistics.mean(rms_col5_jit1)
mean_col5_jit2 = statistics.mean(rms_col5_jit2)
mean_col6_jit1 = statistics.mean(rms_col6_jit1)
mean_col6_jit2 = statistics.mean(rms_col6_jit2)
mean_col7_jit1 = statistics.mean(rms_col7_jit1)
mean_col7_jit2 = statistics.mean(rms_col7_jit2)
mean_col8_jit1 = statistics.mean(rms_col8_jit1)
mean_col8_jit2 = statistics.mean(rms_col8_jit2)

mean_col3_k1 = statistics.mean(rms_col3_k1)
mean_col3_k2 = statistics.mean(rms_col3_k2)
mean_col4_k1 = statistics.mean(rms_col4_k1)
mean_col4_k2 = statistics.mean(rms_col4_k2)
mean_col5_k1 = statistics.mean(rms_col5_k1)
mean_col5_k2 = statistics.mean(rms_col5_k2)
mean_col6_k1 = statistics.mean(rms_col6_k1)
mean_col6_k2 = statistics.mean(rms_col6_k2)
mean_col7_k1 = statistics.mean(rms_col7_k1)
mean_col7_k2 = statistics.mean(rms_col7_k2)
mean_col8_k1 = statistics.mean(rms_col8_k1)
mean_col8_k2 = statistics.mean(rms_col8_k2)

mean_col3_m1 = statistics.mean(rms_col3_m1)
mean_col3_m2 = statistics.mean(rms_col3_m2)
mean_col4_m1 = statistics.mean(rms_col4_m1)
mean_col4_m2 = statistics.mean(rms_col4_m2)
mean_col5_m1 = statistics.mean(rms_col5_m1)
mean_col5_m2 = statistics.mean(rms_col5_m2)
mean_col6_m1 = statistics.mean(rms_col6_m1)
mean_col6_m2 = statistics.mean(rms_col6_m2)
mean_col7_m1 = statistics.mean(rms_col7_m1)
mean_col7_m2 = statistics.mean(rms_col7_m2)
mean_col8_m1 = statistics.mean(rms_col8_m1)
mean_col8_m2 = statistics.mean(rms_col8_m2)

mean_col3_d1 = statistics.mean(rms_col3_d1)
mean_col3_d2 = statistics.mean(rms_col3_d2)
mean_col4_d1 = statistics.mean(rms_col4_d1)
mean_col4_d2 = statistics.mean(rms_col4_d2)
mean_col5_d1 = statistics.mean(rms_col5_d1)
mean_col5_d2 = statistics.mean(rms_col5_d2)
mean_col6_d1 = statistics.mean(rms_col6_d1)
mean_col6_d2 = statistics.mean(rms_col6_d2)
mean_col7_d1 = statistics.mean(rms_col7_d1)
mean_col7_d2 = statistics.mean(rms_col7_d2)
mean_col8_d1 = statistics.mean(rms_col8_d1)
mean_col8_d2 = statistics.mean(rms_col8_d2)

def compare_mean(mean1, mean2):
  if (mean2 > mean1):
    result = "Test set has higher muscle activation result"
  else:
    result = "Normal set has higher muscle activation result"
  return result

print(compare_mean(mean_col3_j1, mean_col3_j2))
print(compare_mean(mean_col4_j1, mean_col4_j2))
print(compare_mean(mean_col5_j1, mean_col5_j2))
print(compare_mean(mean_col6_j1, mean_col6_j2))
print(compare_mean(mean_col7_j1, mean_col7_j2))
print(compare_mean(mean_col8_j1, mean_col8_j2))

print(compare_mean(mean_col3_jit1, mean_col3_jit2))
print(compare_mean(mean_col4_jit1, mean_col4_jit2))
print(compare_mean(mean_col5_jit1, mean_col5_jit2))
print(compare_mean(mean_col6_jit1, mean_col6_jit2))
print(compare_mean(mean_col7_jit1, mean_col7_jit2))
print(compare_mean(mean_col8_jit1, mean_col8_jit2))

print(compare_mean(mean_col3_k1, mean_col3_k2))
print(compare_mean(mean_col4_k1, mean_col4_k2))
print(compare_mean(mean_col5_k1, mean_col5_k2))
print(compare_mean(mean_col6_k1, mean_col6_k2))
print(compare_mean(mean_col7_k1, mean_col7_k2))
print(compare_mean(mean_col8_k1, mean_col8_k2))

print(compare_mean(mean_col3_m1, mean_col3_m2))
print(compare_mean(mean_col4_m1, mean_col4_m2))
print(compare_mean(mean_col5_m1, mean_col5_m2))
print(compare_mean(mean_col6_m1, mean_col6_m2))
print(compare_mean(mean_col7_m1, mean_col7_m2))
print(compare_mean(mean_col8_m1, mean_col8_m2))

print(compare_mean(mean_col3_d1, mean_col3_d2))
print(compare_mean(mean_col4_d1, mean_col4_d2))
print(compare_mean(mean_col5_d1, mean_col5_d2))
print(compare_mean(mean_col6_d1, mean_col6_d2))
print(compare_mean(mean_col7_d1, mean_col7_d2))
print(compare_mean(mean_col8_d1, mean_col8_d2))

x1_j = list_create(len(col1_j1))
x2_j = list_create(len(col1_j2))

x1_jit = list_create(len(col1_jit1))
x2_jit = list_create(len(col1_jit2))

x1_k = list_create(len(col1_k1))
x2_k = list_create(len(col1_k2))

x1_m = list_create(len(col1_m1))
x2_m = list_create(len(col1_m2))

x1_d = list_create(len(col1_d1))
x2_d = list_create(len(col1_d2))
'''
legend_text = ["Jin Normal set", "Jin Test set", "Jitse Normal set", "Jitse Test set",\
                              "Kevin Normal set", "kevin Test set", "Marco Normal set", "Marco Test set",\
                              "Dimitris Normal set", "Dimitris Test set"]
fig1 = plt.figure('Anterior deltoids',figsize=(12,3))
plt.subplot(1,2,1)
plt.plot(x1_j, rms_col3_j1, color='blue')
plt.plot(x2_j, rms_col3_j2, color='red')
plt.plot(x1_jit, rms_col3_jit1, color='chocolate')
plt.plot(x2_jit, rms_col3_jit2, color='maroon')
plt.plot(x1_k, rms_col3_k1, color='limegreen')
plt.plot(x2_k, rms_col3_k2, color='green')
plt.plot(x1_m, rms_col3_m1, color='cyan')
plt.plot(x2_m, rms_col3_m2, color='dodgerblue')
plt.plot(x1_d, rms_col3_d1, color='violet')
plt.plot(x2_d, rms_col3_d2, color='darkviolet')
plt.xlabel("4 repetitions %")
plt.title("Right Anterior deltoids")
plt.legend(legend_text)

#fig2 = plt.figure('Left Anterior deltoids',figsize=(8,6))
plt.subplot(1,2,2)
plt.plot(x1_j, rms_col4_j1, color='blue')
plt.plot(x2_j, rms_col4_j2, color='red')
plt.plot(x1_jit, rms_col4_jit1, color='chocolate')
plt.plot(x2_jit, rms_col4_jit2, color='maroon')
plt.plot(x1_k, rms_col4_k1, color='limegreen')
plt.plot(x2_k, rms_col4_k2, color='green')
plt.plot(x1_m, rms_col4_m1, color='cyan')
plt.plot(x2_m, rms_col4_m2, color='dodgerblue')
plt.plot(x1_d, rms_col4_d1, color='violet')
plt.plot(x2_d, rms_col4_d2, color='darkviolet')
plt.title("Left Anterior deltoids")
plt.legend(legend_text)


fig2 = plt.figure('Pectoralis major',figsize=(12,3))
plt.subplot(1,2,1)
plt.plot(x1_j, rms_col5_j1, color='blue')
plt.plot(x2_j, rms_col5_j2, color='red')
plt.plot(x1_jit, rms_col5_jit1, color='chocolate')
plt.plot(x2_jit, rms_col5_jit2, color='maroon')
plt.plot(x1_k, rms_col5_k1, color='limegreen')
plt.plot(x2_k, rms_col5_k2, color='green')
plt.plot(x1_m, rms_col5_m1, color='cyan')
plt.plot(x2_m, rms_col5_m2, color='dodgerblue')
plt.plot(x1_d, rms_col5_d1, color='violet')
plt.plot(x2_d, rms_col5_d2, color='darkviolet')
plt.title("Left Pectoralis major")
plt.legend(legend_text)

plt.subplot(1,2,2)
plt.plot(x1_j, rms_col6_j1, color='blue')
plt.plot(x2_j, rms_col6_j2, color='red')
plt.plot(x1_jit, rms_col6_jit1, color='chocolate')
plt.plot(x2_jit, rms_col6_jit2, color='maroon')
plt.plot(x1_k, rms_col6_k1, color='limegreen')
plt.plot(x2_k, rms_col6_k2, color='green')
plt.plot(x1_m, rms_col6_m1, color='cyan')
plt.plot(x2_m, rms_col6_m2, color='dodgerblue')
plt.plot(x1_d, rms_col6_d1, color='violet')
plt.plot(x2_d, rms_col6_d2, color='darkviolet')
plt.title("Right Pectoralis major")
plt.legend(legend_text)

fig3 = plt.figure('Triceps brachii',figsize=(12,3))
plt.subplot(1,2,1)
plt.plot(x1_j, rms_col7_j1, color='blue')
plt.plot(x2_j, rms_col7_j2, color='red')
plt.plot(x1_jit, rms_col7_jit1, color='chocolate')
plt.plot(x2_jit, rms_col7_jit2, color='maroon')
plt.plot(x1_k, rms_col7_k1, color='limegreen')
plt.plot(x2_k, rms_col7_k2, color='green')
plt.plot(x1_m, rms_col7_m1, color='cyan')
plt.plot(x2_m, rms_col7_m2, color='dodgerblue')
plt.plot(x1_d, rms_col7_d1, color='violet')
plt.plot(x2_d, rms_col7_d2, color='darkviolet')
plt.title("Left Triceps brachii")
plt.legend(legend_text)

plt.subplot(1,2,2)
plt.plot(x1_j, rms_col8_j1, color='blue')
plt.plot(x2_j, rms_col8_j2, color='red')
plt.plot(x1_jit, rms_col8_jit1, color='chocolate')
plt.plot(x2_jit, rms_col8_jit2, color='maroon')
plt.plot(x1_k, rms_col8_k1, color='limegreen')
plt.plot(x2_k, rms_col8_k2, color='green')
plt.plot(x1_m, rms_col8_m1, color='cyan')
plt.plot(x2_m, rms_col8_m2, color='dodgerblue')
plt.plot(x1_d, rms_col8_d1, color='violet')
plt.plot(x2_d, rms_col8_d2, color='darkviolet')
plt.title("Right Left Triceps brachii")
plt.legend(legend_text)
'''
legend_text1 = legend_text = ["Jin Normal set", "Jin Test set", "Jitse Normal set", "Jitse Test set",\
                              "Kevin Normal set", "kevin Test set", "Marco Normal set", "Marco Test set",\
                              "Dimitris Normal set", "Dimitris Test set"]

# rep x-axis
reps = [1, 2, 3, 4]

# mean velocity
Jin_set1 = [0.41, 0.47, 0.47, 0.45]
Jin_set2 = [0.60, 0.54, 0.52, 0.52]

Jitse_set1 = [0.71, 0.57, 0.61, 0.65]
Jitse_set2 = [0.67, 0.55, 0.49, 0.50]

Kevin_set1 = [0.37, 0.29, 0.33, 0.30]
Kevin_set2 = [0.67, 0.55, 0.49, 0.50]

Marco_set1 = [0.38, 0.34, 0.35, 0.34]
Marco_set2 = [0.41, 0.45, 0.46, 0.37]

Dim_set1 = [0.42, 0.34, 0.36, 0.37]
Dim_set2 = [0.39, 0.39, 0.33, 0.33]


zipped_lists_normal = zip(Jin_set1, Jitse_set1, Kevin_set1, Marco_set1, Dim_set1)
zipped_lists_test = zip(Jin_set2, Jitse_set2, Kevin_set2, Marco_set2, Dim_set2)

# 创建5个新列表，每个新列表包含每个位置的元素
result_lists_normal = [list(elements) for elements in zipped_lists_normal]
result_lists_test = [list(elements) for elements in zipped_lists_test]
result_lists = result_lists_normal + result_lists_test
print(result_lists)
Normal_set = [(a + b + c + d + e) / 5 for a, b, c, d, e in zip(Jin_set1, Jitse_set1, Kevin_set1, Marco_set1, Dim_set1)]
Test_set = [(a + b + c + d + e) / 5 for a, b, c, d, e in zip(Jin_set2, Jitse_set2, Kevin_set2, Marco_set2, Dim_set2)]
data_barbell_compare = [Normal_set, Test_set]

data_barbell = [Jin_set1, Jin_set2, Jitse_set1, Jitse_set2, Kevin_set1, Kevin_set2, Marco_set1, Marco_set2, Dim_set1, Dim_set2]



'''


# 绘制箱线图
fig5 = plt.figure("Average speed of reps between normal set and test set", figsize=(8,6))
barbell_speed_compare = plt.boxplot(data_barbell_compare, vert=True, patch_artist=True)

# 设置箱线图的颜色和标签
colors = ['red', 'darkviolet']
labels = ["Normal set 1st rep", "Test set 1st rep", "Normal set 2nd rep", "Test set 2nd rep",\
          "Normal set 3rd rep", "Test set 3rd rep", "Normal set 4th rep", "Test set 4th rep",]

for box, color, label in zip([Normal_set['boxes'], Test_set['boxes']], colors, labels):
    box.set(facecolor=color)

# 显示方差
for i, box in enumerate(plt.boxplot(data_barbell_compare)['boxes']):
    box.set_gapcolor('lightblue')
    variance = np.var(data_barbell_compare[i])
    plt.text(i + 1, max(data_barbell_compare[i]) + 0.5, f'Var: {variance:.2f}', horizontalalignment='center')

# 设置图表标题和标签
plt.legend([Normal_set['boxes'][0], Test_set['boxes'][0],Normal_set['boxes'][1], Test_set['boxes'][1],\
            Normal_set['boxes'][2], Test_set['boxes'][2], Normal_set['boxes'][3], Test_set['boxes'][3]],\
           labels, loc='upper right')
plt.title('Box Plot with Variance')
plt.xlabel('Groups')
plt.ylabel('Average barbell speed of one repetition')
'''

def find_mean(input_data):
  input_data_mean = []
  for i in range(len(input_data)):
    input_data_mean.append(input_data.mean())
    i = i + 1
  return input_data_mean

rms_j1_mean = find_mean(rms_j1)
rms_j2_mean = find_mean(rms_j2)
rms_jit1_mean = find_mean(rms_jit1)
rms_jit2_mean = find_mean(rms_jit2)
rms_k1_mean = find_mean(rms_k1)
rms_k2_mean = find_mean(rms_k2)
rms_m1_mean = find_mean(rms_m1)
rms_m2_mean = find_mean(rms_m2)
rms_d1_mean = find_mean(rms_d1)
rms_d2_mean = find_mean(rms_d2)

legend_text = ['Normal set', 'Encouragement set', 'Normal set average', 'Encouragement set average']

fig1 = plt.figure(figsize=(8,6))
plt.plot(x1_j, rms_j1, color='dodgerblue')
plt.plot(x2_j, rms_j2, color='red')
plt.plot(x1_j, rms_j1_mean, color='cyan', linestyle='--')
plt.plot(x2_j, rms_j2_mean, color='orange', linestyle='--')
plt.xlabel("% 4 reps")
plt.ylabel("Activation([mv]")
plt.title("Cumulative Muscle Activations: Normal VS Encouragement")
plt.legend(legend_text)

fig2 = plt.figure(figsize=(8,6))
plt.plot(x1_jit, rms_jit1, color='dodgerblue')
plt.plot(x2_jit, rms_jit2, color='red')
plt.plot(x1_jit, rms_jit1_mean, color='cyan', linestyle='--')
plt.plot(x2_jit, rms_jit2_mean, color='orange', linestyle='--')
plt.xlabel("% 4 reps")
plt.ylabel("Activation([mv]")
plt.title("Cumulative Muscle Activations: Normal VS Encouragement")
plt.legend(legend_text)

fig3 = plt.figure(figsize=(8,6))
plt.plot(x1_k, rms_k1, color='dodgerblue')
plt.plot(x2_k, rms_k2, color='red')
plt.plot(x1_k, rms_k1_mean, color='cyan', linestyle='--')
plt.plot(x2_k, rms_k2_mean, color='orange', linestyle='--')
plt.xlabel("% 4 reps")
plt.ylabel("Activation([mv]")
plt.title("Cumulative Muscle Activations: Normal VS Encouragement")
plt.legend(legend_text)

fig4 = plt.figure(figsize=(8,6))
plt.plot(x1_m, rms_m1, color='dodgerblue')
plt.plot(x2_m, rms_m2, color='red')
plt.plot(x1_m, rms_m1_mean, color='cyan', linestyle='--')
plt.plot(x2_m, rms_m2_mean, color='orange', linestyle='--')
plt.xlabel("% 4 reps")
plt.ylabel("Activation([mv]")
plt.title("Cumulative Muscle Activations: Normal VS Encouragement")
plt.legend(legend_text)

fig5 = plt.figure(figsize=(8,6))
plt.plot(x1_d, rms_d1, color='dodgerblue')
plt.plot(x2_d, rms_d2, color='red')
plt.plot(x1_d, rms_d1_mean, color='cyan', linestyle='--')
plt.plot(x2_d, rms_d2_mean, color='orange', linestyle='--')
plt.xlabel("% 4 reps")
plt.ylabel("Activation([mv]")
plt.title("Cumulative Muscle Activations: Normal VS Encouragement")
plt.legend(legend_text)

# 绘制箱线图
fig6 = plt.figure("Average speed of reps", figsize=(8,6))

meanline_props = dict(color='limegreen', linestyle='--', linewidth=2)

medianline_props = dict(color='darkviolet', linestyle='-', linewidth=2)

barbell_speed = plt.boxplot(result_lists, vert=True, patch_artist=True,  meanline=True, showmeans=True, meanprops=meanline_props, medianprops=medianline_props)

for median in barbell_speed['medians']:
    median.set(visible=False)

# 设置箱线图的颜色和标签

colors = ['dodgerblue','dodgerblue','dodgerblue','dodgerblue', 'red','red','red','red']
labels = ["Normal set rep1", "Normal set rep2", "Normal set rep3", "Normal set rep4",\
                              "Encouragement set rep1", "Encouragement set rep2", "Encouragement set rep3", "Encouragement set rep4"]



for box, color, label in zip(barbell_speed['boxes'], colors, labels):
    box.set(facecolor=color)

# 显示方差
for i, box in enumerate(plt.boxplot(result_lists)['boxes']):
    box.set_gapcolor('lightblue')
    variance = np.var(result_lists[i])
    plt.text(i + 1, max(result_lists[i]) + 0.5, f'Var: {variance:.2f}', horizontalalignment='center')

# 设置图表标题和标签
plt.legend([barbell_speed['boxes'][0], barbell_speed['boxes'][1], barbell_speed['boxes'][2],\
            barbell_speed['boxes'][3], barbell_speed['boxes'][4], barbell_speed['boxes'][5],\
            barbell_speed['boxes'][6], barbell_speed['boxes'][7]], labels, loc='upper right')

plt.title('Box Plot with variance for average barbell speed in different groups')
plt.xlabel('Repetitions')
plt.ylabel('Average barbell speed of one repetition')
# Show the plot
plt.show()