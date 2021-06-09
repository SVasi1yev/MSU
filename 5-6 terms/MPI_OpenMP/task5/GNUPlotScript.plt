set terminal png size 1600,900
set output "times.png"
set ylabel "Время, сек"
set xlabel "Число ядер"
plot "time.txt" u 1:2 title "1024x1024" w l lc 1, "time.txt" u 1:3 title "2048x2048" w l lc 2, "time.txt" u 1:4 title "4096x4096" w l lc 3
set output "speedup.png"
set ylabel "Ускорение"
set xlabel "Число ядер"
plot "speedup.txt" u 1:2 title "1024x1024" w l lc 1, "speedup.txt" u 1:3 title "2048x2048" w l lc 2, "speedup.txt" u 1:4 title "4096x4096" w l lc 3
set output "effect.png"
set ylabel "Эффективность"
set xlabel "Число ядер"
plot "effect.txt" u 1:2 title "1024x1024" w l lc 1, "effect.txt" u 1:3 title "2048x2048" w l lc 2, "effect.txt" u 1:4 title "4096x4096" w l lc 3
set output "file_time.png"
set ylabel "Время ввода/вывода, сек"
set xlabel "Число ядер"
plot "file_time.txt" u 1:2 title "1024x1024" w l lc 1, "file_time.txt" u 1:3 title "2048x2048" w l lc 2, "file_time.txt" u 1:4 title "4096x4096" w l lc 3
