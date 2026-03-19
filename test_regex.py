import re

text1 = "Es importante no utilizar nombres propios, lugares ni marcas; manténgalos tal y como aparecen en el texto original o en su forma romanizada. La traducción debe ser concisa. Estás rodeado por una oscuridad perfecta..."
text2 = "SON IMPORTANTES: No se deben traducir los nombres propios, las ubicaciones ni las marcas; manténgalos tal y como aparecen en el texto original o en su forma romanizada. Haga que la traducción sea concisa. Estas estrellas tienden a ser más pequeñas..."
text3 = "Es importante no utilizar nombres propios, lugares ni marcas; manténgalos tal y como aparecen en el texto original o en su forma romanizada. Haga que la traducción sea concisa. El texto se refiere al espacio que se crea..."
text4 = "IMPORTANTE: No incluya nombres propios, ubicaciones ni marcas; manténgalos tal y como aparecen en el texto original o en su forma romanizada. Haga que la traducción sea concisa. Es posible que encuentre un producto..."

pattern = r"(?i)(?:es\s+|son\s+)?importante[s]?[\s:]*no\s+.*?concisa\."

for i, t in enumerate([text1, text2, text3, text4], 1):
    res = re.sub(pattern, "", t).strip()
    print(f"Test {i}: {res}")

