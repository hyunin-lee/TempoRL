from symfit import parameters, variables, sin, cos, Fit

class linear_fitting_solver:
    def __init__(self):
        self.x, self.y = variables('x, y')
        self.a, self.b = parameters('a,b')
        self.model_dict =  {self.y: self.a * self.x + self.b}


class Fourier_fitting_solver:
    def __init__(self,order):
        self.x, self.y = variables('x, y')
        self.w, = parameters('w')
        self.coeff = {}
        self.model_dict = {self.y: self.fourier_series(self.x, f=self.w, n=order)}

    # def addattr(self,x,val):
    #     self.__dict__[x]=val

    def addattr2(self,x,val):
        self.coeff[x]=val

    def fourier_series(self,x, f, n):
        """
        Returns a symbolic fourier series of order `n`.

        :param n: Order of the fourier series.
        :param x: Independent variable
        :param f: Frequency of the fourier series
        """
        # Make the parameter objects for all the terms
        a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
        for a in cos_a :
            self.addattr2(str(a),a)
        sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
        for b in sin_b :
            self.addattr2(str(b), b)
        # Construct the series
        series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                         for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
        return series

class cos_fitting_solver:
    def __init__(self,order):
        self.x, self.y = variables('x, y')
        self.w, = parameters('w')
        self.model_dict = {self.y: self.cos_series(self.x, f=self.w, n=order)}

    def cos_series(self,x, f, n):
        """
        Returns a symbolic fourier series of order `n`.

        :param n: Order of the fourier series.
        :param x: Independent variable
        :param f: Frequency of the fourier series
        """

        a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
        # sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
        # series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
        #                  for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
        series = a0 + sum(ai * cos(i * f * x)
                         for i, ai in enumerate(cos_a, start=1))
        return series

class sin_fitting_solver:
    def __init__(self,order):
        self.x, self.y = variables('x, y')
        self.w, = parameters('w')
        self.model_dict = {self.y: self.sin_series(self.x, f=self.w, n=order)}

    def sin_series(self,x, f, n):
        """
        Returns a symbolic fourier series of order `n`.

        :param n: Order of the fourier series.
        :param x: Independent variable
        :param f: Frequency of the fourier series
        """

        # a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
        self.b0, *sin_b = parameters(','.join(['b{}'.format(i) for i in range(0, n + 1)]))
        # series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
        #                  for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
        series = self.b0 + sum(bi * sin(i * f * x)
                         for i, bi in enumerate(sin_b, start=1))
        return series

    def sin_series_wo_constant(self,x, f, n):
        """
        Returns a symbolic fourier series of order `n`.

        :param n: Order of the fourier series.
        :param x: Independent variable
        :param f: Frequency of the fourier series
        """

        # a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
        sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
        # series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
        #                  for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
        series = sum(bi * sin(i * f * x) for i, bi in enumerate(sin_b, start=1))
        return series
