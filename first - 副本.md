
记$$
m_{ij}=\frac{1}{2}\begin{bmatrix} 1+\frac{n_j \cos \theta_j}{n_i \cos \theta_i} &  1-\frac{n_j \cos \theta_j}{n_i \cos \theta_i}\\ 1-\frac{n_j \cos \theta_j}{n_i \cos \theta_i} & 1+\frac{n_j \cos \theta_j}{n_i \cos \theta_i} \\\end{bmatrix}
$$

则$$
m_{i1}=\frac{1}{2}\begin{bmatrix} 1+\frac{n_1 \cos \theta_1}{n_i \cos \theta_i} &  1-\frac{n_1 \cos \theta_1}{n_i \cos \theta_i}\\ 1-\frac{n_1 \cos \theta_1}{n_i \cos \theta_i} & 1+\frac{n_1 \cos \theta_1}{n_i \cos \theta_i} \\\end{bmatrix}
$$
$$
m_{12}=\frac{1}{2}\begin{bmatrix} 1+\frac{n_2}{n_1} &  1-\frac{n_2}{n_1} \\ 1-\frac{n_2}{n_1} & 1+\frac{n_2}{n_1} \\\end{bmatrix}
$$
$$
m_{20}=\frac{1}{2}\begin{bmatrix} 1+\frac{n_0}{n_2} & 1-\frac{n_0}{n_2} \\ 1-\frac{n_0}{n_2} & 1+\frac{n_0}{n_2} \\\end{bmatrix}
$$

记
$$
m_i=\begin{bmatrix} e^{- \mathrm{i} \delta} & 0 \\ 0 &  e^{ \mathrm{i} \delta}\\  \end{bmatrix}
$$

则
$$
m_1=\begin{bmatrix} -\mathrm{i}  & 0 \\ 0 &  \mathrm{i} \\  \end{bmatrix}
$$

$$
m_2=\begin{bmatrix} -\mathrm{i}  & 0 \\ 0 &  \mathrm{i} \\  \end{bmatrix}
$$


$$
\begin{bmatrix} E_1^+ \\ E_1^- \\\end{bmatrix}=M\begin{bmatrix} E_N^+ \\ 0 \\\end{bmatrix}
$$

$$
M=m_{i1} \cdot m_1 \cdot m_{12} \cdot m_2 \cdot m_{20}=\begin{bmatrix} M_{11} & M_{12} \\ M_{21} & M_{22}  \\\end{bmatrix}
$$

$$R=|r|^2=\left|\frac{M_{21}}{M_{11}}\right|^2$$

$$=\left|\frac{2 n_i n_2^2 \cos \theta_i + (n_1 n_2^2 - n_1^2 n_0 + n_0 n_1 n_2 + n_1^2 n_2 \cos \theta_1)}{2n_i n_2^2 \cos \theta_i +(n_1^2 n_2 + n_0 n_1^2)\cos \theta_1}\right|^2.$$

----
$$
\begin{aligned}
R&=|r|^2=\left|\frac{M_{21}}{M_{11}}\right|^2 \\
&=\left| \frac{-\frac{n_2}{n_1}+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}}{\frac{n_2}{n_1}+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}} \right|^2
\end{aligned}
$$

$$
R=|r|^2=\left|\frac{M_{21}}{M_{11}}\right|^2=\left| \frac{-\frac{n_2}{n_1}+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}}{\frac{n_2}{n_1}+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}} \right|^2
$$

----
$$
R=|r|^2=\left|\frac{M_{21}}{M_{11}}\right|^2=\left| \frac{1- (\frac{n_1}{n_2})^2 \cdot \frac{n_0}{n_i} \cdot \frac{\cos \theta_1}{ \cos \theta_i}}{1+ (\frac{n_1}{n_2})^2 \cdot \frac{n_0}{n_i} \cdot \frac{\cos \theta_1}{ \cos \theta_i}} \right|^2
$$

----
比对周期
$$
1.R=\left| \frac{\frac{n_2}{n_1} - \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}}{\frac{n_2}{n_1}+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}} \right|^2
$$
$$
2.R=\left| \frac{(\frac{n_2}{n_1})^2 - \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2} \cdot \frac{n_1}{n_2}}{(\frac{n_2}{n_1})^2+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}\cdot \frac{n_1}{n_2}} \right|^2
$$
$$
3.R=\left| \frac{(\frac{n_2}{n_1})^3 - \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2} \cdot (\frac{n_1}{n_2})^2}{(\frac{n_2}{n_1})^3+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}\cdot (\frac{n_1}{n_2})^2} \right|^2
$$
$$
N.R=\left| \frac{(\frac{n_2}{n_1})^N - \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2} \cdot (\frac{n_1}{n_2})^{N-1}}{(\frac{n_2}{n_1})^N+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}\cdot (\frac{n_1}{n_2})^{N-1}} \right|^2
$$
$$
=\left| \frac{(\frac{n_1}{n_2})^{2N} \cdot \frac{n_0}{n_i}\cdot \frac{\cos \theta_1}{ \cos \theta_i} - 1}{(\frac{n_1}{n_2})^{2N} \cdot \frac{n_0}{n_i}\cdot \frac{\cos \theta_1}{ \cos \theta_i} + 1} \right|^2
$$

$$
\begin{aligned}
R &=\left| \frac{(\frac{n_2}{n_1})^N - \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2} \cdot (\frac{n_1}{n_2})^{N-1}}{(\frac{n_2}{n_1})^N+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}\cdot (\frac{n_1}{n_2})^{N-1}} \right|^2 \\
&=\left| \frac{(\frac{n_2}{n_1})^N - \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2} \cdot (\frac{n_1}{n_2})^{N-1}}{(\frac{n_2}{n_1})^N+ \frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2}\cdot (\frac{n_1}{n_2})^{N-1}} \right|^2
\end{aligned}
$$

----
$$
\begin{aligned}
M_{11}&=2(1-\frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2})(1- \frac{n_2}{n_1}) \\&+ (1+\frac{n_0}{n_2})(1+\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{-2 \mathrm{i}\delta} \\&+ (1-\frac{n_0}{n_2})(1-\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{2 \mathrm{i}\delta}
\end{aligned}
$$

$$
\begin{aligned}
M_{21}&=2(1-\frac{n_2}{n_1} \cdot \frac{n_0}{n_2})(1- \frac{n_1\cos \theta_1}{n_i \cos \theta_i}) \\&+ (1+\frac{n_0}{n_2})(1-\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{-2 \mathrm{i}\delta} \\&+ (1-\frac{n_0}{n_2})(1+\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{2 \mathrm{i}\delta}
\end{aligned}
$$

$$
R=|r|^2=\left|\frac{2(1-\frac{n_2}{n_1} \cdot \frac{n_0}{n_2})(1- \frac{n_1\cos \theta_1}{n_i \cos \theta_i}) + (1+\frac{n_0}{n_2})(1-\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{-2 \mathrm{i}\delta} + (1-\frac{n_0}{n_2})(1+\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{2 \mathrm{i}\delta}}{2(1-\frac{n_1\cos \theta_1}{n_i \cos \theta_i} \cdot \frac{n_0}{n_2})(1- \frac{n_2}{n_1}) + (1+\frac{n_0}{n_2})(1+\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{-2 \mathrm{i}\delta} + (1-\frac{n_0}{n_2})(1-\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{2 \mathrm{i}\delta}}\right|^2
$$

$$
R=|r|^2=\left|\frac{M_{21}&=2(1-\frac{n_2}{n_1} \cdot \frac{n_0}{n_2})(1- \frac{n_1\cos \theta_1}{n_i \cos \theta_i})}{2(1-\frac{n_2}{n_1} \cdot \frac{n_0}{n_2})(1- \frac{n_1\cos \theta_1}{n_i \cos \theta_i})+ (1+\frac{n_0}{n_2})(1-\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{-2 \mathrm{i}\delta}+ (1-\frac{n_0}{n_2})(1+\frac{n_1\cos \theta_1}{n_i \cos \theta_i})(1+\frac{n_2}{n_1})e^{2 \mathrm{i}\delta}}|^2
$$

----

$$
m_{ij} = \frac{1}{2n_i \cos \theta_i} \frac{1}{n_i \cos \theta_i + n_j \cos \theta_j} \begin{bmatrix}
1 & \frac{n_i \cos \theta_i - n_j \cos \theta_j}{n_i \cos \theta_i + n_j \cos \theta_j} \\
\frac{n_i \cos \theta_i - n_j \cos \theta_j}{n_i \cos \theta_i + n_j \cos \theta_j} & 1
\end{bmatrix}
$$
