Descriptions 

- `cp_simple.py`: the implementation of cp with additional item (considering second term). This method is a bit complex, so now we simplify it by assuming M(x) is free of theta

- `cp_simple1wx.py`: the implementation of cp without considering second term but calculating the conditional expectation.  Conditioning on x is not necessary in my simulation so I can ignore it. Also now (2024-09-16), I only consider the case when $x$ is categorical, if it is continuous, it can be more complex.