import datetime
import pprint
# from Rnx2Pos import Rnx2Pos as RP
from matplotlib import pyplot as plt

from geodetic_time_serie import geodetic_time_serie as GDT
from pandas import Period as Pd
import datetime as dt

# Reshala=RP()
# Reshal=Reshala.read_dirs(input_nav=r"C:\Users\Admin\Downloads\tetsRNX\board",input_base=r"C:\Users\Admin\Downloads\tetsRNX\NOVM",input_rover=r"C:\Users\Admin\Downloads\tetsRNX\NSK1")
# Reshala.start(timeint='120',path_rtklib=r"C:\Users\Admin\Downloads\tetsRNX\rnx2rtkp.exe",rtklib_conf=r"C:\Users\Admin\Downloads\tetsRNX\rnx2rtkp.conf",output_dir=r"C:\Users\Admin\Downloads\tetsRNX\output")
# pprint.pprint(Reshala.start(timeint='120',path_rtklib=r"C:\Users\Admin\Downloads\tetsRNX\rnx2rtkp.exe",rtklib_conf=r"C:\Users\Admin\Downloads\tetsRNX\rnx2rtkp.conf",output_dir=r"C:\Users\Admin\Downloads\tetsRNX\output"))
# file=open('nomm.txt')
# while True:
# считываем строку
#  line = file.readline()
#  module_name = Word(alphas + '_')
# full_module_name = module_name + ZeroOrMore('.' + module_name)
#   import_as = Optional('|' + module_name)
#  parse_module = '12367M002' + full_module_name + import_as
#  print(parse_module.parseString(line))
#  if not line:
#     break
#  # выводим строку
count = 1000

a = GDT.test_coordinate_time_series_generator(count, 1000, 10000)

# print(a.lombescargle(axe_column="latitude(deg)",date_column="lGPSTtime",comb="day_year"))
# delt=dt.timedelta(minutes=2)
# a=GDT.pos2geodetic_time_serie(posfilepath=r"C:\Users\Admin\Downloads\tetsRNX\output",solution_selection_type="max_ratio",freqw=delt,sol_freq=dt.timedelta(hours=10))

print(a)

a = GDT(a)

time_list = []
for t in range(count):
    _date = datetime.date(2015, 1, 14)
    _time = datetime.time(12, 30)
    now_date = datetime.datetime.combine(_date, _time)
    new_date = datetime.timedelta(weeks=t)
    new_date = new_date + now_date
    time_list.append(new_date)


a.drop('Epoch', axis=1, inplace=True)



a.loc[:, "epoch"] = time_list
# print(a.lombescargle(axe_column="X", date_column="epoch", comb="day_year"))
a = a.set_index('epoch')


plt.plot(a.X)
plt.grid()
plt.show()



a.drop(['Y', 'Z'], axis=1, inplace=True)

# a=a.outliers_filtering(axe='latitude(deg)',type= 'iqr')

# print(a.outliers_filtering(axe='latitude(deg)',type='iqr'))
# print(a.outliers_filtering(axe='latitude(deg)',type='median'))


# print(a.lombescargle(axe_column="X",date_column="Epoch",comb='hour_day'))

print(a)
