# Derivative directions and diagnostics

Training derivative labels use 16 frozen covariance-informed directions. The
independent metric `E_J` averages errors on these finite probes and is not an
operator-norm estimate. The paper therefore also reports a common-basepoint
diagnostic comparing fixed-direction JVP errors, ambient same-residual VJP
errors and VJP errors after the Matérn covariance action used by Landweber.

The Direction-Informed Subspace (DIS) study is retained in the private
experiment catalogue but is not part of the frozen paper products or public
paper pipeline.
