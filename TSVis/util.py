
def hex2rgb(color):
    h = color.lstrip('#')
    return [int(h[i:i+2], 16)/ 255.0 for i in (0, 2, 4)]

FRAME_COLOR = [0.5,0.5,0.5,1]
BLUE_COLOR = hex2rgb('#4E79A7')
BLACK_COLOR = hex2rgb('#000000')
MARGIN_X = 0.02
MARGIN_Y = 0.05
tableau10 = ["#4e79a7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]
tableau10 = [hex2rgb(t) for t in tableau10]
tableau10light = ['#AEC7E8','#FFBB78','#ff9896','#9edae5','#98DF8A','#dbdb8d','#c5b0d5','#f7b6d2','#c49c94','#c7c7c7']
tableau10light = [hex2rgb(t) for t in tableau10light]