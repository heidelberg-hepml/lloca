Tips and tricks for stable LLoCa training
=========================================

Regularizing input particles
-----------------------------

Most particles in an LHC setting can be considered massless as the typical energy scale is much larger than the mass of single particles.
When operating on four-momenta, the particle masses are often assumed to be zero or can be modified due to numerical underflows,
leading to particles with zero-norm four-momenta. Since zero-norm vectors can cause downstream instabilities, we enforce a minimal
particle mass :math:`m_\epsilon` by increasing the energy of all input particles :math:`p_i` as
$$E' = \\sqrt{m_{\\epsilon}^2 + E^2},$$
hence :math:`m'^2 = m_\epsilon^2 + m^2`.
We use :math:`m_\epsilon=10^{-5}` and :math:`m_\epsilon=5\cdot 10^{-3}` for the amplitude regression and tagging experiments, respectively.
In our tests, we observe a large plateau of stable :math:`m_\epsilon` values and we ultimately select the smallest viable value.
Performance typically degrades for :math:`m_\epsilon\geq 1`. The units are in standardized space, menaning that the regularization is applied
after rescaling the input four-momenta to :math:`\mathcal{O}(1)` numbers.

Predicting stable local frames
-------------------------------

We carefully design the construction of local frames to avoid numerical instabilities during training.
Here, we describe the main aspects for the MLP Frames-Net but they apply also to the LGATr and PELICAN Frames-Nets.
The vectors :math:`v_{i,k}`, needed to construct a local frame for each particle :math:`i`, are predicted using
$$v_{i,k} = \\sum_{j=1}^N \\mathrm{softmax}\\Big( \\varphi_k(s_i, s_j, \\langle p_i, p_j\\rangle )\\Big) \\frac{p_i + p_j}{\\|p_i+p_j\\| + \\epsilon} \\quad  \\text{ for } k = 0,1,2.$$

We have identified several techniques to avoid numerical instabilities already at initialization and throughout the training:

1. we include a :math:`\mathrm{softmax}` operation to prevent any negative-norm vectors in the process, leaving the transformation into spacelike vectors for three out of four vectors to the Gram-Schimdt orthonormalization and the local frame construction;
2. the usage of regulator masses to enforce a lower limit on the norms of the input particles :math:`p_i` already described above;
3. we multiply by :math:`p_i+p_j` instead of simply :math:`p_i` because :math:`p_i+p_j` is typically not strongly boosted;
4. we perform an additional normalization step on the final predictions: $$v_{i,k} \\gets v_{i,k} \\Big/ \\sqrt{\\sum_{j=1}^N \\| v_{i,k}\\|^2}.$$

Regularizing predicted vectors
-------------------------------

Even with the above precautions, in rare cases the predicted vectors :math:`v_{i,k}` might produce ill-defined local frames.
We apply the following regularizations to ensure that the training remains stable.

**NB:** These regularizations break Lorentz-equivariance. They are meant to be applied in rare cases
or only at initialization such that the training remains stable. The application of these regularizations to a large
fraction of particles in the batch indicates that the numerics of the input should be improved.

We verify that :math:`v_0` is not lightlike, as this would lead to an ill-defined boost :math:`B`.
We check that
$$\\|v_0\\| < \\epsilon_{\\text{lightlike}} \\qquad\\text{with}\\qquad \\epsilon_{\\text{lightlike}}=10^{-16}.$$
If this condition is not satisfied, we replace :math:`v_0` with :math:`v_0 + \epsilon_\text{lightlike}\delta`, with :math:`\delta`
a timelike noise four-vector constructed by sampling the components of a vector from
a Gaussian distribution, taking the absolute value, and setting the energy
component to two times the spatial norm.

Then, the :math:`\mathrm{SO}(3)` Gram-Schmidt orthogonalization used to construct the
rotation matrix :math:`R` requires linearly independent vectors. We numerically control
this condition by verifying that
$$\\|\\vec w_1 \\times \\vec w_2\\| < \\epsilon_{\\text{collinear}}.$$
If a regularization is needed, we replace both vectors with
$$\\vec w_1 + \\epsilon_{\\text{collinear}}\\vec \\delta_1 \\qquad\\text{and}\\qquad \\vec w_2 + \\epsilon_{\\text{collinear}}\\vec \\delta_2,$$
where :math:`\vec \delta_{1,2}` are random normal directions, i.e.~:math:`\delta_i \sim\mathcal{N}(0,1)`.
This choice of regularization allows for safe prediction of local frames also in
the edge case of :math:`N<3` input particles. For both :math:`N=1` and :math:`N=2`, the cross product
between :math:`\vec w_1` and :math:`\vec w_2`` would be zero; the regularization ensures that
the vectors are linearly independent.
In our studies, we use :math:`\epsilon_\text{collinear}=10^{-16}`, and we do not observe regularization throughout trainings besides a few
occurrences at initialization, validating the numerical stability of our orthogonalization method.

Both regularizations are implemented in the LLoCa package. The default values are set to the minimum epsilon which can be represented
in the chosen floating point precision. User-defined values can be set in the ``framesnet`` configs.

Gram-Schimdt orthonormalization in 3D
-------------------------------------

For completeness, we report here the numerical details of the :math:`\mathrm{SO}(3)` Gram-Schmidt orthonormalization used in the local frame construction.
:math:`\mathrm{GS3}(\vec w_1, \vec w_2)` takes the space components of the predicted vectors after applying the learned boost.
The resulting vectors :math:`\vec e_1`, :math:`\vec e_2`, and :math:`\vec e_3` are orthonormal, i.e. they satisfy :math:`\vec e_i \cdot \vec e_j = \delta_{ij}`.

**GS(3):**

.. math::
    \begin{align}
    \operatorname{norm}(\vec w) = \frac{\vec w}{\|\vec w\| + \epsilon} \qquad
    \vec e_1 &= \operatorname{norm}\left(\vec w_1 \right),\notag \\
    \vec e_2 &= \operatorname{norm}\left(\vec w_2 - \vec e_1 (\vec w_2 \cdot \vec e_1)\right), \notag\\
    \vec e_3 &= \vec e_1 \times \vec e_2.
    \label{eq:gram-schmidt}
    \end{align}

We use :math:`\epsilon = 10^{-15}`, and :math:`\times` denotes the cross product.
