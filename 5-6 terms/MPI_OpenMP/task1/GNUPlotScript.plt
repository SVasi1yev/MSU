set terminal png size 640,480
set output 'plot.png'
set ylabel "Время, сек"
set xtics ("ijk" 0,"ikj" 1,"kij" 2,"jik" 3,"jki" 4,"kji" 5)
set title "1000x1000x1000"
plot "Time.dat" w l lc 1
