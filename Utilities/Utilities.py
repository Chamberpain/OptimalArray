import numpy as np
import scipy.sparse

def make_R(Cov,H_holder,noise_factor):
	idx,dummy = H_holder.base_return()
	noise_divider = H_holder.return_noise_divider()
	idx = np.sort(idx)
	out = Cov[idx,idx]*noise_factor/noise_divider
	# out = noise_factor*np.ones([1,len(idx)])/noise_divider
	return scipy.sparse.diags(np.ravel(out))

def make_Gain(H,Cov,R):
	denom = H.dot(Cov).dot(H.T)+R
	# assert (abs(denom - denom.T) > 1**-10).nnz == 0
	inv_denom = np.linalg.inv(denom.todense())
	# assert (abs(inv_denom - inv_denom.T) < 1**-10).all()

	Cov = scipy.sparse.csr_matrix(Cov)
	H = scipy.sparse.csr_matrix(H)
	inv_denom = scipy.sparse.csr_matrix(inv_denom)
	transformed_inv = (H.T).dot(inv_denom).dot(H)
	# assert (abs(transformed_inv - transformed_inv.T) > 1**-10).nnz == 0	
	KH = Cov.dot(transformed_inv)
	return KH

def make_GainFactor(H,Cov,R):
	KH = make_Gain(H,Cov,R)
	factor = scipy.sparse.eye(Cov.shape[0])-KH
	return factor

def make_CovSubtract(Cov,H_holder,noise_factor=2):
	R = make_R(Cov,H_holder,noise_factor)
	KH = make_Gain(H_holder.return_H(),Cov,R)
	cov_subtract = KH.dot(Cov)
	return cov_subtract


def make_P_hat(Cov,H_holder,noise_factor=2):
	R = make_R(Cov,H_holder,noise_factor)
	gainFactor = make_GainFactor(H_holder.return_H(),Cov,R)
	# assert (abs(gainFactor - gainFactor.T) > 1**-10).nnz == 0
	# assert all(gainFactor.diagonal() >= 0)
	p_hat = gainFactor.dot(Cov)
	# assert (abs(p_hat - p_hat.T) > 1**-10).nnz == 0
	# assert all(p_hat.diagonal() >= 0)

	return p_hat

def get_index_of_first_eigen_vector(p_hat,trans_geo):
	eigs = scipy.sparse.linalg.eigs(p_hat,k=1)
	e_vec = eigs[1][:,-1]
	e_vec_sum = sum(abs(x) for x in np.split(e_vec,len(trans_geo.variable_list)))
	idx = e_vec_sum.tolist().index(e_vec_sum.max())
	return idx,e_vec