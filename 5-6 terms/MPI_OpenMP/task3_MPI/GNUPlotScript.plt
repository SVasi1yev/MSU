set terminal png size 1600,900
set output "max_times.png"
set ylabel "Время, сек"
set xlabel "Число процессов"
plot "times.dat" u 1:2 title "max" w l lc 1
set output "sum_times.png"
set ylabel "Время, сек"
set xlabel "Число процессов"
plot "times.dat" u 1:3 title "sum" w l lc 1
