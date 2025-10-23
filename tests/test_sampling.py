from bblean.similarity import jt_stratified_sampling
from bblean.fingerprints import make_fake_fingerprints

def test_stratified():
    fps = make_fake_fingerprints(10, seed=134)
    out = jt_stratified_sampling(fps, 8)
    print(out)

test_stratified()
