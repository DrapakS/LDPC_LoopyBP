import numpy as np
import numpy.testing as npt
import sys
import unittest
import ldpc


class TestLDPC(unittest.TestCase):

    def test_python_version(self):
        npt.assert_allclose(sys.version_info[0], 3)

    def test_generator_matrix_creation(self):
        n = 80
        m = 25
        H = np.random.randint(2, size=(m,n))
        H1 = H.copy()
        G, ind = ldpc.make_generator_matrix(H)
        res = H.dot(G) % 2
        npt.assert_allclose(res, 0)
        npt.assert_almost_equal(np.linalg.norm(G[ind, :] - np.eye(n-m)), 0)
        npt.assert_almost_equal(np.linalg.norm(H - H1), 0)

    def test_degenerate_check_matrix(self):
        n = 80
        m = 25
        H = np.random.randint(2, size=(m,n))
        ind = np.random.permutation(m)
        H[ind[0], :] = H[ind[1], :]
        H[ind[2], :] = H[ind[3], :]
        try:
            ldpc.make_generator_matrix(H)
        except ldpc.DegenerateMatrixError:
            pass
        else:
            raise Exception()

    def test_update_messages_e_to_h(self):
        mu_h_to_e = np.array([[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3],
                              [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3],
                              [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3]])
        mu_h_to_e = np.moveaxis(np.tile(mu_h_to_e, (4, 1, 1)), 0, -1)
        mu_e_to_h = np.zeros((3, 7, 4))
        true_mu_e_to_h = np.zeros_like(mu_e_to_h)
        true_mu_e_to_h[:, :, 0] = np.array([[0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0.01219512, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0]])
        true_mu_e_to_h[:, :, 2] = np.array([[0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0.01219512, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0]])
        beliefs = np.zeros((7, 4))
        true_beliefs = np.zeros_like(beliefs)
        true_beliefs[:, 0] = np.array([0, 0, 0, 0, 0.00136986, 0, 0])
        true_beliefs[:, 2] = np.array([0, 0, 0, 0, 0.00136986, 0, 0])

        ldpc.update_messages_e_to_h_and_beliefs(mu_e_to_h, beliefs, mu_h_to_e, 0.1, 4, np.array([0, 1]), [1],
                                           np.array([0, 2]))
        npt.assert_array_almost_equal(mu_e_to_h, true_mu_e_to_h)
        npt.assert_array_almost_equal(beliefs, true_beliefs)

    def test_update_messages_h_to_e(self):
        mu_h_to_e = np.zeros((3, 7, 4))
        mu_e_to_h = np.array([[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3],
                              [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3],
                              [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3]])
        mu_e_to_h = np.moveaxis(np.tile(mu_e_to_h, (4, 1, 1)), 0, -1)

        true_mu_h_to_e = np.zeros_like(mu_h_to_e)
        true_mu_h_to_e[:,:,0] = np.array([[0, 0, 0, 0, 0, 0, 0],
                                          [0, 0.58, 0, 0, 0.56, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0]])
        true_mu_h_to_e[:,:,2] = np.array([[0, 0, 0, 0, 0, 0, 0],
                                          [0, 0.42, 0, 0, 0.44, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0]])
        s = np.array([[0, 0, 1, 1],
                      [1, 0, 0, 1],
                      [0, 1, 0, 0]])

        ldpc.update_messages_h_to_e(mu_h_to_e, mu_e_to_h, s, 1, np.array([1, 3, 4]), np.array([1, 4]), np.array([0, 2]))
        npt.assert_array_almost_equal(mu_h_to_e, true_mu_h_to_e)

    def test_easy_decoding(self):
        H = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.int8)

        e = np.zeros((16,3), dtype=np.int32)
        for i_trial in np.arange(3):
            t = np.random.randint(16)
            e[t, i_trial] = 1

        s = H.dot(e) % 2
        [hat_e, results] = ldpc.decode(s, H, 0.1, display=False)
        npt.assert_array_almost_equal(results['status'], 0)

    def test_hard_decoding(self):
        H = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.int8)

        e = np.zeros((16, 3), dtype=np.int32)
        for i_trial in np.arange(3):
            t = np.random.permutation(16)
            e[t[:10], i_trial] = 1

        s = H.dot(e) % 2
        [hat_e, results] = ldpc.decode(s, H, 0.62, display=False)
        npt.assert_array_less(0, results['status'])


if __name__ == '__main__':
    unittest.main()
