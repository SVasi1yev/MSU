set terminal png size 800,450

set output "time_plot.png"
set ylabel "Time, sec"
set xtics ("0.5k" 0,"1k" 1,"1.5k" 2, "2k" 3, "2.5k" 4)
plot "format_output.dat" u 1:2 title "32-ijk" w l lc 1, \
     "format_output.dat" u 1:3 title "32-ikj" w l lc 2, \
     "format_output.dat" u 1:4 title "opt-ijk" w l lc 3
set output "cycle_plot.png"
set ylabel "Cycles"
set xtics ("0.5k" 0,"1k" 1,"1.5k" 2, "2k" 3, "2.5k" 4)
plot "format_output.dat" u 1:5 title "32-ijk" w l lc 1, \
     "format_output.dat" u 1:6 title "32-ikj" w l lc 2, \
     "format_output.dat" u 1:7 title "opt-ijk" w l lc 3
set output "L1_misses_plot.png"
set ylabel "L1 cache misses"
set xtics ("0.5k" 0,"1k" 1,"1.5k" 2, "2k" 3, "2.5k" 4)
plot "format_output.dat" u 1:8 title "32-ijk" w l lc 1, \
     "format_output.dat" u 1:9 title "32-ikj" w l lc 2, \
     "format_output.dat" u 1:10 title "opt-ijk" w l lc 3
set output "L2_misses_plot.png"
set ylabel "L2 cache misses"
set xtics ("0.5k" 0,"1k" 1,"1.5k" 2, "2k" 3, "2.5k" 4)
plot "format_output.dat" u 1:11 title "32-ijk" w l lc 1, \
     "format_output.dat" u 1:12 title "32-ikj" w l lc 2, \
     "format_output.dat" u 1:13 title "opt-ijk" w l lc 3
set output "FLOPs_plot.png"
set ylabel "FLOPs"
set xtics ("0.5k" 0,"1k" 1,"1.5k" 2, "2k" 3, "2.5k" 4)
plot "format_output.dat" u 1:14 title "32-ijk" w l lc 1, \
     "format_output.dat" u 1:15 title "32-ikj" w l lc 2, \
     "format_output.dat" u 1:16 title "opt-ijk" w l lc 3
