import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoiseSchedule:
    '''
        s rappresenta un offset, poichè il coseno parte quasi piatto orizzontalmente
        al primo step il sistema aggiungerebbe una quantità troppo piccola
        portando il sistema ad errori di calcolo o divisioni che tendono all'infinito
        per questo si fissa il valore a 0.008 per partire da un punto meno piatto
    '''
    def __init__(self, L, s=0.008, device=device): 
        self.L=L
        t=torch.linspace(0.0, L, L+1, device=device)/L
        a=torch.cos((t+s)/(1+s)*torch.pi/2)**2
        a=a/a[0]
        self.beta=(1-a[1:]/a[:-1]).clip(0.0, 0.99) #poiché il valore di beta potrebbe diventare più grande di 1 facciamo un clip
                                                   #in modo che non superi il suo valore limite
        self.alpha=torch.cumprod(1.0-self.beta, dim=0)
        self.one_minus_beta=1-self.beta
        self.one_minus_alpha=1-self.alpha
        self.sqrt_alpha=torch.sqrt(self.alpha)
        self.sqrt_beta=torch.sqrt(self.beta)
        self.sqrt_1_alpha=torch.sqrt(self.one_minus_alpha)
        self.sqrt_1_beta=torch.sqrt(self.one_minus_beta)

    def __len__(self):
        return self.L