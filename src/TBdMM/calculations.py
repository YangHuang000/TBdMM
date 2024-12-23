#!/usr/bin/env python

import numpy as np
from ase import io

reference_data = {
    'Sc': {'d_cen': 1.7032, 'rd': 0.9409409409409409, 'dij': 3.27, 'Wd': 5.999737383},
    'Ti': {'d_cen': 1.3243, 'rd': 0.7907907907907907, 'dij': 2.91, 'Wd': 6.741977763},
    'V':  {'d_cen': 0.5548, 'rd': 0.6906906906906907, 'dij': 2.70, 'Wd': 6.977374918},
    'Cr': {'d_cen': -0.0876,'rd': 0.6306306306306306, 'dij': 2.56, 'Wd': 7.209708128},
    'Mn': {'d_cen': -0.6036,'rd': 0.5905905905905906, 'dij': 2.48, 'Wd': 7.084501084},
    'Fe': {'d_cen': -0.9278,'rd': 0.6206206206206206, 'dij': 2.53, 'Wd': 7.112329237},
    'Co': {'d_cen': -1.5905,'rd': 0.5605605605605606, 'dij': 2.45, 'Wd': 6.696932181},
    'Ni': {'d_cen': -1.6686,'rd': 0.5205205205205206, 'dij': 2.49, 'Wd': 5.354092732},
    'Cu': {'d_cen': -2.6521,'rd': 0.4904904904904905, 'dij': 2.58, 'Wd': 4.244656352},
    'Y':  {'d_cen': 2.2707, 'rd': 1.2512512512512513, 'dij': 3.58, 'Wd': 7.873690234},
    'Zr': {'d_cen': 1.6485, 'rd': 1.0710710710710711, 'dij': 3.20, 'Wd': 8.905095875},
    'Nb': {'d_cen': 0.6878, 'rd': 0.9409409409409409, 'dij': 2.98, 'Wd': 8.976212519},
    'Mo': {'d_cen': -0.011, 'rd': 0.8608608608608609, 'dij': 2.84, 'Wd': 9.057322241},
    'Ru': {'d_cen': -1.6305,'rd': 0.7507507507507507, 'dij': 2.72, 'Wd': 8.066324527},
    'Rh': {'d_cen': -2.0874,'rd': 0.7207207207207207, 'dij': 2.72, 'Wd': 7.39255913},
    'Pd': {'d_cen': -2.087, 'rd': 0.6706706706706707, 'dij': 2.79, 'Wd': 5.679312123},
    'Ag': {'d_cen': -4.1421,'rd': 0.6606606606606606, 'dij': 2.94, 'Wd': 4.503883909},
    'Ta': {'d_cen': 1.1906, 'rd': 1.021021021021021, 'dij': 2.98, 'Wd': 10.89131422},
    'W':  {'d_cen': 0.1732, 'rd': 0.9409409409409409, 'dij': 2.86, 'Wd': 10.73729515},
    'Re': {'d_cen': -1.0633,'rd': 0.8808808808808809, 'dij': 2.75, 'Wd': 10.99527175},
    'Os': {'d_cen': -1.9693,'rd': 0.8508508508508509, 'dij': 2.72, 'Wd': 10.63999091},
    'Ir': {'d_cen': -2.6636,'rd': 0.8208208208208209, 'dij': 2.74, 'Wd': 9.50224736},
    'Pt': {'d_cen': -2.6369,'rd': 0.7907907907907907, 'dij': 2.82, 'Wd': 7.638177581},
    'Au': {'d_cen': -3.6577,'rd': 0.7607607607607607, 'dij': 2.95, 'Wd': 5.83120117}
}

def get_hopping(ele_1, ele_2, ele_site, idx_1, idx_2, rd_1, rd_2, orb_1, orb_2, dij, R):
    l, m, n = R / np.linalg.norm(R)
    hopping_int = 0.0
    if idx_1 != idx_2:
        if dij > 5.5:
            hopping_int = 0
        elif (orb_1 == 's') and (orb_2 == 's'):
            v_ss_sigma = 7.62 * (-1.4) * (1 / dij**2)
            hopping_int = v_ss_sigma
        elif (orb_1 == 's') and (orb_2 in ['dxy','dyz','dxz','dz2','dx2-z2']):
            v_sd_sigma = 7.62 * (-3.16) * (rd_2**1.5 / dij**3.5)
            if orb_2 == 'dxy':
                hopping_int = (3**0.5) * l * m * v_sd_sigma
            elif orb_2 == 'dyz':
                hopping_int = (3**0.5) * m * n * v_sd_sigma
            elif orb_2 == 'dxz':
                hopping_int = (3**0.5) * l * n * v_sd_sigma
            elif orb_2 == 'dz2':
                hopping_int = (n**2 - (l**2 + m**2) / 2) * v_sd_sigma
            elif orb_2 == 'dx2-z2':
                hopping_int = (3**0.5 / 2) * (l**2 - m**2) * v_sd_sigma
        elif (orb_1 in ['dxy','dyz','dxz','dz2','dx2-z2']) and (orb_2 == 's'):
            v_sd_sigma = 7.62 * (-3.16) * (rd_1**1.5 / dij**3.5)
            if orb_1 == 'dxy':
                hopping_int = (3**0.5) * l * m * v_sd_sigma
            elif orb_1 == 'dyz':
                hopping_int = (3**0.5) * m * n * v_sd_sigma
            elif orb_1 == 'dxz':
                hopping_int = (3**0.5) * l * n * v_sd_sigma
            elif orb_1 == 'dz2':
                hopping_int = (n**2 - (l**2 + m**2) / 2) * v_sd_sigma
            elif orb_1 == 'dx2-z2':
                hopping_int = (3**0.5 / 2) * (l**2 - m**2) * v_sd_sigma
        elif (orb_1 in ['dxy','dyz','dxz','dz2','dx2-z2']) and (orb_2 in ['dxy','dyz','dxz','dz2','dx2-z2']):
            v_dd_sigma = 7.62 * (-16.2) * (rd_1**1.5 * rd_2**1.5 / dij**5)
            v_dd_pi = 7.62 * (8.75) * (rd_1**1.5 * rd_2**1.5 / dij**5)
            v_dd_delta = 0
            if (orb_1 == 'dxy') and (orb_2 == 'dxy'):
                hopping_int = 3*l**2*m**2*v_dd_sigma + (l**2 + m**2 - 4*l**2*m**2)*v_dd_pi + (n**2 + l**2*m**2)*v_dd_delta
            elif ((orb_1 == 'dxy') and (orb_2 == 'dyz')) or ((orb_2 == 'dxy') and (orb_1 == 'dyz')):
                hopping_int = l*n*(3*m**2*v_dd_sigma + (1 - 4*m**2)*v_dd_pi + (m**2 - 1)*v_dd_delta)
            elif ((orb_1 == 'dxy') and (orb_2 == 'dxz')) or ((orb_2 == 'dxy') and (orb_1 == 'dxz')):
                hopping_int = m*n*(3*l**2*v_dd_sigma + (1 - 4*l**2)*v_dd_pi + (l**2 - 1)*v_dd_delta)
            elif ((orb_1 == 'dxy') and (orb_2 == 'dz2')) or ((orb_2 == 'dxy') and (orb_1 == 'dz2')):
                hopping_int = (3**0.5)*l*m*((n**2 - 0.5*(l**2 + m**2))*v_dd_sigma - 2*n**2*v_dd_pi + 0.5*(1 + n**2)*v_dd_delta)
            elif ((orb_1 == 'dxy') and (orb_2 == 'dx2-z2')) or ((orb_2 == 'dxy') and (orb_1 == 'dx2-z2')):
                hopping_int = l*m*(l**2 - m**2)*(1.5*v_dd_sigma - 2*v_dd_pi + 0.5*v_dd_delta)
            elif (orb_1 == 'dyz') and (orb_2 == 'dyz'):
                hopping_int = 3*n**2*m**2*v_dd_sigma + (n**2 + m**2 - 4*n**2*m**2)*v_dd_pi + (l**2 + n**2*m**2)*v_dd_delta
            elif ((orb_1 == 'dyz') and (orb_2 == 'dxz')) or ((orb_2 == 'dyz') and (orb_1 == 'dxz')):
                hopping_int = m*l*(3*n**2*v_dd_sigma + (1 - 4*n**2)*v_dd_pi + (n**2 - 1)*v_dd_delta)
            elif ((orb_1 == 'dyz') and (orb_2 == 'dz2')) or ((orb_2 == 'dyz') and (orb_1 == 'dz2')):
                hopping_int = (3**0.5)*m*n*((n**2 - 0.5*(l**2 + m**2))*v_dd_sigma + (l**2 + m**2 - n**2)*v_dd_pi - 0.5*(l**2 + m**2)*v_dd_delta)
            elif ((orb_1 == 'dyz') and (orb_2 == 'dx2-z2')) or ((orb_2 == 'dyz') and (orb_1 == 'dx2-z2')):
                hopping_int = m*n*(1.5*(l**2 - m**2)*v_dd_sigma - (1 + 2*(l**2 - m**2))*v_dd_pi + (1 + 0.5*(l**2 - m**2))*v_dd_delta)
            elif (orb_1 == 'dxz') and (orb_2 == 'dxz'):
                hopping_int = 3*l**2*n**2*v_dd_sigma + (l**2 + n**2 - 4*l**2*n**2)*v_dd_pi + (m**2 + l**2*n**2)*v_dd_delta
            elif ((orb_1 == 'dxz') and (orb_2 == 'dx2-z2')) or ((orb_2 == 'dxz') and (orb_1 == 'dx2-z2')):
                hopping_int = n*l*(1.5*(l**2 - m**2)*v_dd_sigma + (1 - 2*(l**2 - m**2))*v_dd_pi - (1 - 0.5*(l**2 - m**2))*v_dd_delta)
            elif (orb_1 == 'dz2') and (orb_2 == 'dz2'):
                hopping_int = (n**2 - 0.5*(l**2 + m**2))**2*v_dd_sigma + 3*n**2*(l**2 + m**2)*v_dd_pi + 0.75*(l**2 + m**2)**2*v_dd_delta
            elif ((orb_1 == 'dz2') and (orb_2 == 'dx2-z2')) or ((orb_2 == 'dz2') and (orb_1 == 'dx2-z2')):
                hopping_int = (3**0.5)*(l**2 - m**2)*(0.5*(n**2 - 0.5*(l**2 + m**2))*v_dd_sigma - n**2*v_dd_pi + 0.25*(1 + n**2)*v_dd_delta)
            elif (orb_1 == 'dx2-z2') and (orb_2 == 'dx2-z2'):
                hopping_int = 0.75*(l**2 - m**2)**2*v_dd_sigma + (l**2 + m**2 - (l**2 - m**2)**2)*v_dd_pi + (n**2 + 0.25*(l**2 - m**2)**2)*v_dd_delta
    else:
        if (orb_1 == orb_2) and (orb_1 != 's'):
            hopping_int = reference_data[ele_1]['d_cen'] - reference_data[ele_site]['d_cen']
        elif (orb_1 == orb_2) and (orb_1 == 's'):
            hopping_int = 0 - reference_data[ele_site]['d_cen']
        else:
            hopping_int = 0
    return hopping_int

def moment_n(H, n):
    """
    Compute H^n, then sum first 5 diagonal elements.
    """
    Hn = np.linalg.matrix_power(H, n)
    mn = 0.0
    for i in range(5):
        mn += Hn[i][i]
    return mn

def _TBdMM_core(image, site_index, n):
    """
    Build Hamiltonian H, then compute m_n = trace of H^n (only first 5 diag).
    Return (m_n, H).
    """
    site_index = len(image) * 4 + site_index
    copy_image = image.copy()
    copy_image = copy_image.repeat((3,3,1))
    unrelax_image = copy_image

    Natm = 50
    all_dij = unrelax_image.get_all_distances()
    nbr_dis = all_dij[int(site_index)]
    all_R = unrelax_image.get_positions()[:,None,:] - unrelax_image.get_positions()[None,:,:]

    sort_index = np.argsort(nbr_dis)[:Natm]
    sort_element = np.array(unrelax_image.get_chemical_symbols())[sort_index]
    sort_rd = [reference_data[ele]['rd'] for ele in sort_element]

    H = np.zeros((Natm*6, Natm*6))
    orb_list = ['dxy','dyz','dxz','dz2','dx2-z2','s']

    for i in range(Natm):
        for j in range(Natm):
            for ii in range(6):
                for jj in range(6):
                    H[6*i+ii, 6*j+jj] = get_hopping(
                        sort_element[i],
                        sort_element[j],
                        sort_element[0],
                        i, j,
                        sort_rd[i], sort_rd[j],
                        orb_list[ii], orb_list[jj],
                        all_dij[sort_index[i], sort_index[j]],
                        all_R[sort_index[i], sort_index[j]]
                    )

    m_n = moment_n(H, n)
    return m_n, H

def TBdMM_1(image, site_index):
    """
    Special case: n=1 uses the formula m1 = sqrt(12*m2)*bulk_ratio,
    but we still return the Hamiltonian for reference if needed.
    """
    site_element = image[site_index].symbol
    bulk_ratio = reference_data[site_element]['d_cen'] / reference_data[site_element]['Wd']

    m2, H = _TBdMM_core(image, site_index, 2)  # get m2 from n=2
    m1 = (12 * m2)**0.5 * bulk_ratio
    return m1, H  # also return the same H (though it's from n=2 build)

def TBdMM(image, site_index, n):
    """
    Public interface: return (mn, H).
    If n<1 or not an int, print error and return None, None.
    """
    if not isinstance(n, int) or n < 1:
        print("The order n is invalid.")
        return None, None

    if n == 1:
        m1, H = TBdMM_1(image, site_index)
        return m1, H
    else:
        m_n, H = _TBdMM_core(image, site_index, n)
        return m_n, H