# GMML
This repository is the Python implementation of paper _"[Robust Beamforming for RIS-aided Communications: Gradient-based Manifold Meta Learning](https://ieeexplore.ieee.org/document/10623434)"_, which has been accepted by _IEEE Transactions on Wireless Communications 2024_

A simplified version, titled _"[Energy-efficient Beamforming for RIS-aided Communications: Gradient Based Meta Learning](https://ieeexplore.ieee.org/document/10622978)"_ and with manifold learning technique removed, has been accepted for _2024 IEEE International Conference on Communications (ICC)_.

## Blog
English version : [Click here](https://zhuanlan.zhihu.com/p/695011497).

Chinese version : [Click here](https://zhuanlan.zhihu.com/p/686734331).

## Files in this repo
`main.py`: The main function. Can be directly run to get the results.

`utils.py`: This file contains the util functions, including the intialization functions and calculation function of spectral efficiency. It also contains definition of system params.

`net.py`: This file defines and declares the neural networks and their params.

`TWC_Paper.pdf`: This file is the PDF file of the paper.

## Reference
Should you find this work beneficial, **kindly grant it a star**!

To follow our research, **please consider citing**:

F. Zhu et al., "Robust Beamforming for RIS-Aided Communications: Gradient-Based Manifold Meta Learning," in _IEEE Transactions on Wireless Communications_, vol. 23, no. 11, pp. 15945-15956, Nov. 2024.

X. Wang, F. Zhu, Q. Zhou, Q. Yu, C. Huang, A. Alhammadi, Z. Zhang, C. Yuen, and M. Debbah, "Energy-efficient Beamforming for RISs-aided Communications: Gradient Based Meta Learning," in _Proc. of the 2024 IEEE International Conference on Communications (ICC)_, Jun. 9, 2024, pp. 3464-3469.


```bibtex

@ARTICLE{Zhu2024GMML,
  author={Zhu, Fenghao and Wang, Xinquan and Huang, Chongwen and Yang, Zhaohui and Chen, Xiaoming and Alhammadi, Ahmed and Zhang, Zhaoyang and Yuen, Chau and Debbah, Mérouane},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Robust Beamforming for RIS-aided Communications: Gradient-based Manifold Meta Learning}, 
  year={2024},
  volume={23},
  number={11},
  pages={15945-15956},
  keywords={Reconfigurable intelligent surfaces;meta learning;manifold learning;gradient;beamforming},
  doi={10.1109/TWC.2024.3435023}}

@inproceedings{Wang2024EnergyEfficient,
  author = {X. Wang and F. Zhu and Q. Zhou and Q. Yu and C. Huang and A. Alhammadi and Z. Zhang and C. Yuen and M. Debbah},
  title = {{Energy-efficient Beamforming for RISs-aided Communications: Gradient Based Meta Learning}},
  booktitle = {Proc. of the 2024 IEEE International Conference on Communications (ICC)},
  year = {2024},
  date = {Jun. 9},
  pages = {3464-3469}
}

```
## More than GMML...
We are excited to announce a novel method that utilizes linear approximations of **ODE-based neural networks** to optimize sum rate in beamforming in mmWave MIMO systems. 

Compared to baseline, it only uses **1.6\% of time** to optimize and achieves a **significantly stronger robustness**! 

See [GLNN](https://github.com/tp1000d/GLNN) for more information!

## Star History

<a href="https://star-history.com/#FenghaoZhu/GMML&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FenghaoZhu/GMML&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FenghaoZhu/GMML&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=FenghaoZhu/GMML&type=Date" />
 </picture>
</a>
