# 2-Dimensional Ring

In this two-dimensional example we use a typical example
for multi-channeling where standard `VEGAS` fails. In this example we train both, the multi-channel weights $\alpha_i(x,\theta)$ 
as well as the conditional normalizing flow $H(x,\varphi\vert i)$ which yields the importance weight $h(x,\varphi\vert i)$. 
In principle, we can combine this numeric importance weight with an additional weight $g_i(x)$ coming from an analytic phase-space mapping
$G_i(x)$ to a total importance weight

$$ q(x,\varphi\vert i) = h(x,\varphi\vert i)\cdot g_i(x).$$

<div align="center">
<img src="circle.png" width="400">
</div>

## Function

We consider the overlap of a gaussian ring and a gaussian 'line' distribution, which is defined by:

```math
P_\text{multi}=\frac{1}{2}P_\text{ring}+\frac{1}{2}P_\text{line}\,
```

with

```math
P_\text{ring}(x_1,x_2) = N_0\,\mathrm{exp}\left(-\frac{1}{2\sigma_0^2}(\sqrt{x_1^2+x_2^2}-r_0)^2\right),\qquad\qquad\qquad\quad
```
```math
P_\text{line}(x_1,x_2) = N_1\,\mathrm{exp}\left(-\frac{1}{2\sigma_1^2}(\tilde{x}_1-\mu_1)^2\right)\mathrm{exp}\left(-\frac{1}{2\sigma_2^2}(\tilde{x}_2-\mu_2)^2\right),
```

where $N_0$ and $N_1$ are chosen such that each distribution is normalized individually and

```math
\tilde{x}_1=\frac{1}{\sqrt{2}}\left(x_1-x_2\right)\,,\qquad \tilde{x}_2=\frac{1}{\sqrt{2}}\left(x_1+x_2\right)\,.
```

## Training

Commands for `mcw`:

```python
# train the multi-channel weights only with fixed analytic mappings (optinally adding arguments, see --help)
python train_mcw.py (--arg ARG)
```

Commands for `flow`:

```python
# train the flow only with fixed weights (optinally adding arguments, see --help)
python train_flow.py (--arg ARG)
```

Commands for `mc-flow`:

```python
# train the flow simultanously with the multi-channel weights (optinally adding arguments, see --help)
python train_mcflow.py (--arg ARG)
```

Commands for `map-mc-flow` (does not work yet!):

```python
# train the flow simultanously with the multi-channel weights 
# including analytic mappings (optinally adding arguments, see --help)
python train_map_mcflow.py (--arg ARG)
```

## Analytic remapping

### Mapping I

In order to improve the integration of the two-modal distribution we introduce a channel-mapping for each of the two distributions. First, we consider the sub-density

$$
P_\text{line}(x_1,x_2) = N_1\,\mathrm{exp}\left(-\frac{1}{2\sigma_1^2}(\tilde{x}_1-\mu_1)^2\right)\mathrm{exp}\left(-\frac{1}{2\sigma_2^2}(\tilde{x}_2-\mu_2)^2\right),
$$

with

$$
N_1=\frac{1}{\sqrt{2\pi\sigma_1^2}} \frac{1}{\sqrt{2\pi\sigma_2^2}}.
$$

The sub-integral then reads in Cartesian coordinates

$$
I_1=\int\limits_{-\infty}^{\infty}\mathrm{d} x_1 \int\limits_{-\infty}^{\infty}\mathrm{d} x_2\ P_\text{line}(x_1,x_2).
$$

We first perform a change of variables 

$$\mathbf{x}\to \mathbf{y}$$ 

and its inverse as

$$
y_1=\frac{1}{\sqrt{2}}(x_1-x_2), \quad y_2=\frac{1}{\sqrt{2}}(x_1+x_2),\quad
$$

$$
x_1=\frac{1}{\sqrt{2}}(y_1+y_2), \quad x_2=\frac{1}{\sqrt{2}}(-y_1+y_2).
$$

And consequently, the Jacobian determinant is simply given by 

$$\det J_1=1.$$ 

Thus, the integral reads

$$
I_1=\int\limits_{-\infty}^{\infty}\mathrm{d} y_1 \int\limits_{-\infty}^{\infty}\mathrm{d} y_2\ P_\text{line}(x_1,x_2)\Big\vert_{\mathbf{x}=\mathbf{x}(\mathbf{y})}
$$

Next, we map out the peak-structures in each direction by using again a Cauchy distribution. Hence, we define the mapping 

$$\mathbf{y}\to\mathbf{z}$$ 

and its inverse as

$$
 z_1=\frac{1}{\pi}\arctan\left(\frac{y_1-\mu_1}{\gamma_1}\right)+\frac{1}{2}, \quad z_2=\frac{1}{\pi}\arctan\left(\frac{y_2-\mu_1}{\gamma_2}\right)+\frac{1}{2},
$$

$$
y_1=\mu_1+\gamma_1\tan\left[\pi\left(z_1-\frac{1}{2}\right)\right], \quad y_2=\mu_2+\gamma_2\tan\left[\pi\left(z_2-\frac{1}{2}\right)\right],
$$

Hence, the Jacobian determinant is given by

$$
\det J_2 = \left\vert\frac{\partial\mathbf{z}}{\partial\mathbf{y}}\right\vert=\frac{1}{\pi\gamma_1\left[1+\left(\frac{y_1-\mu_1}{\gamma_1}\right)^2\right]}\times \frac{1}{\pi\gamma_2\left[1+\left(\frac{y_2-\mu_2}{\gamma_2}\right)^2\right]}.
$$

Consequently, the reparametrized integral reads

$$
I_1=N_1\int\limits_{0}^{1}\mathrm{d} z_1 \int\limits_{0}^{1}\mathrm{d} z_2\left.\frac{\mathrm{exp}\left(-\frac{1}{2\sigma_1^2}(y_1-\mu_1)^2\right)}{\frac{1}{\pi\gamma_1\left[1+\left(\frac{y_1-\mu_1}{\gamma_1}\right)^2\right]}}\times \frac{\mathrm{exp}\left(-\frac{1}{2\sigma_2^2}(y_2-\mu_2)^2\right)}{\frac{1}{\pi\gamma_1\left[1+\left(\frac{y_2-\mu_2}{\gamma_2}\right)^2\right]}}\right\vert_{\mathbf{y}=\mathbf{y}(\mathbf{z})}.
$$

### Mapping II

Next, we consider the circular part of the density which is given by

$$
P_\text{ring}(x_1,x_2) = N_0\,\mathrm{exp}\left(-\frac{1}{2\sigma_0^2}\left(\sqrt{x_1^2+x_2^2}-r_0\right)^2\right).
$$

with

$$
N_0=\left[2\pi\left(\sigma_0^2\,e^{-\frac{r_0^2}{2\sigma_0^2}}+\sqrt{\frac{\pi}{2}}\,r_0\,\sigma_0\left(1+\text{erf}\left[\frac{r_0}{\sqrt{2}\sigma_0}\right]\right)\right)\right]^{-1}.
$$

The sub-integral is then given by

$$
I_2=\int\limits_{-\infty}^{\infty}\mathrm{d} x_1 \int\limits_{-\infty}^{\infty}\mathrm{d} x_2\ P_\text{ring}(x_1,x_2)\,.
$$

We first go to polar coordinates and define the change of variables 

$$\mathbf{x}\to\mathbf{r}$$

and its inverse as

$$
r=\sqrt{x_1^2+x_2^2}, \quad \theta=\arctan\frac{x_2}{x_1},
$$

$$
x_1=r\cos\theta, \qquad x_2=r\sin\theta.\quad\quad
$$

Consequently, the Jacobian determinant is simply given by 

$$\det J_1=r.$$ 

Thus, the integral reads in polar coordinates

$$
I_1\int\limits_{0}^{\infty}\mathrm{d} r\ r \int\limits_{0}^{2\pi}\mathrm{d} \theta\ P_\text{ring}(x_1,x_2)\Big\vert_{\mathbf{x}=\mathbf{x}(\mathbf{r})}\,.
$$

Finally, we map out the peak-structure in the radial direction using again a Cauchy distribution and a simple linear transformation for the angle $\theta$. Hence, we define the mapping 

$$\mathbf{r}\to\mathbf{z}$$ 

and its inverse as

$$
z_1=\frac{1}{\pi}\arctan\left(\frac{r-r_0}{\gamma_0}\right)+C_0, \quad z_2=\frac{\theta}{2\pi},
$$

$$
r=r_0+\gamma_0\tan\left[\pi\left(z_1-C_0\right)\right], \quad \theta=2\pi z_2,
$$

where the constant 

$$C_0=\frac{1}{\pi}\arctan\left(\frac{r_0}{\gamma_0}\right)$$

ensures that $r>0$. Thus the Jacobian determinant is given by

$$
\det J_2 = \left\vert\frac{\partial\mathbf{z}}{\partial\mathbf{r}}\right\vert
=\frac{1}{2\pi}\times\frac{1}{\pi\gamma_0\left[1+\left(\frac{r-r_0}{\gamma_0}\right)^2\right]}.
$$

Consequently, the integral reads

$$
I_2=N_0\int\limits_{0}^{1}\mathrm{d} z_1 \int\limits_{0}^{1}\mathrm{d} z_2\left.\frac{r\ \mathrm{exp}\left(-\frac{1}{2\sigma_0^2}(r-r_0)^2\right)}{\frac{1}{2\pi^2\gamma_0\left[1+\left(\frac{r-r_0}{\gamma_0}\right)^2\right]}}\right\vert_{\mathbf{r}=\mathbf{r}(\mathbf{z})}.
$$
   
   
