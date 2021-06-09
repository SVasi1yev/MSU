set terminal png size 1600,900
set output "times1.png"
set ylabel "Время, сек"
set xlabel "Число ядер"
plot "time.txt" u 1:2 title "512x512" w l lc 1, "time.txt" u 1:3 title "1024x1024" w l lc 2, "time.txt" u 1:4 title "2048x2048" w l lc 3
set output "times2.png"
set ylabel "Время, сек"
set xlabel "Число ядер"
plot "time.txt" u 1:5 title "4096x4096" w l lc 4, "time.txt" u 1:6 title "4096x1024" w l lc 5, "time.txt" u 1:7 title "1024x4096" w l lc 6
set output "accel1.png"
set ylabel "Ускорение"
set xlabel "Число ядер"
plot "uskor.txt" u 1:2 title "512x512" w l lc 1, "uskor.txt" u 1:3 title "1024x1024" w l lc 2, "uskor.txt" u 1:4 title "2048x2048" w l lc 3
set output "accel2.png"
set ylabel "Ускорение"
set xlabel "Число ядер"
plot "uskor.txt" u 1:5 title "4096x4096" w l lc 4, "uskor.txt" u 1:6 title "4096x1024" w l lc 5, "uskor.txt" u 1:7 title "1024x4096" w l lc 6
set output "effect1.png"
set ylabel "Эффективность"
set xlabel "Число ядер"
plot "effect.txt" u 1:2 title "512x512" w l lc 1, "effect.txt" u 1:3 title "1024x1024" w l lc 2, "effect.txt" u 1:4 title "2048x2048" w l lc 3
set output "effect2.png"
set ylabel "Эффективность"
set xlabel "Число ядер"
plot "effect.txt" u 1:5 title "4096x4096" w l lc 4, "effect.txt" u 1:6 title "4096x1024" w l lc 5, "effect.txt" u 1:7 title "1024x4096" w l lc 6
