import time
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from utils_extract_pipeline import *
import base64

prompt = """
    GOAL: From the image, extract and structure:
    - headline (string or "NA")
    - subheadline (string or "NA")
    - author (string or "NA")
    - content (string)

    IMPORTANT:
    - Only process meaningful journalistic articles.
    - Exclude short notices like dates, weather, ads, announcements.
    - If irrelevant for media analysis, skip.
    - If any field is missing, write "NA".

    RETURN FORMAT:
    Strict CSV:
    "headline";"subheadline";"author";"content"
    "El loco del martillo";"NA";"La Seño María";"Hoy en día, uno pensaría que..."
    "Contento por fin de cuarentena";"Habla Trome";"Ismael Lazo, Vecino de San Luis";"Estoy feliz porque..."

    RULES:
    - No explanations, extra text, or commentary.
    - Use double quotes for fields.
    - Use semicolons (`;`) as separators.
    - Replace internal semicolons with commas.
    - Each row is one article; first row is always the header.

    CONTEXT:
    You are an expert in analyzing and structuring newspaper content. Accuracy is critical. CSV must be precise.
    """

prompt_image = """
            You are an expert OCR system. Extract ALL text content from this newspaper image with perfect accuracy.

            CRITICAL REQUIREMENTS:
            - Read every single word, number, date, and punctuation mark visible in the image
            - The text is in SPANISH - preserve all Spanish accents, tildes, and special characters (ñ, á, é, í, ó, ú, ü)
            - Preserve the original text layout and structure (headlines, paragraphs, columns)
            - Maintain proper spacing between words and sentences
            - Include ALL content: headlines, subheadings, body text, captions, advertisements, page numbers, dates
            - Handle multiple columns by reading left-to-right, top-to-bottom within each column
            - Preserve special characters, accents, and non-English text exactly as shown
            - Do NOT skip any text, even if partially obscured or small
            - Do NOT add explanations, interpretations, or markdown formatting
            - Do NOT summarize or paraphrase - extract the exact text as written
            
            Return ONLY the raw extracted text content, preserving the natural reading flow of the newspaper.
            
            WARNING: If you return anything other than raw text (explanations, apologies, formatting, etc.), 
            the entire OCR pipeline will fail and all downstream processing will be corrupted. 
            Your response must contain ONLY the extracted text - nothing else.
            """

questions = [
    """E  ASCER OOO]  [ay     ¿CM ASER LA ACTIVIDAD TUWO UNA CAÍDA 0d TPL  Venta  de vehículds'e  yen espera repuntar en  junio apoyada por la reconstrucción  Retroceso. li res parado Pre od racha positiva de ho mueves cociremiros, Maya sde arrartrería el estero de 1 Mio contero. Lempaza de ca mictra y pue lapa parte he tercenest cia pres pena Por oa.  sn das A e Dente aporta bel 3518, Lo verca de mdao do trata Patatas de pri demo Bolitas da peores abel premoe Js em dato, las dencia Drs de ena podr ateo da pra tiran is de e Every, deta na Ca Mco YI% e ac), ve ja da tao aras A eri del Pe, _ Jaqrevidense, Eibar AS Pd medio has reas rg región, A fell 9 sets e pe Es ds pue a jefa O, Pret aa esti A me pac der pre,  "Verds de ir e e veras hon, 5er aire Lima, quee pdrusho ds de mps, 10 de legua or quer carta A ON                    Fat ta seo Tina, Aya + qua Vaya ro regenta us dd ate, edo dar desde qe aser, ib Lis d26 Mar. direis aomridado Led 9 e par dt, rod me        A tá Le el preso ies esa (en daran el ¡teporieeracos de lata, Cat Qee, eat ue abel tao Dades tubo tas corpo Joe zaz el anna Y al terco A Pia ventas eo Úlcdao e, Alcor e eepore de dela acción:  Cmucirione a=harop to           Cueas  12,654  rotos Uma y Pin Races oras A, pios arepa de dee     A A ida e 0 0 ta a A A AS - a ser Ventas de vebicutos principales marcas "era dernpamba 608 Seen Tarma rta e: A al Cemagace perrelzacor ade li A                              Pos citó ed a beta or 0d es  podre els iaa. mp eltacode Hymstad, a  A ha reco car dea mues Dr     OTROS OO" Cota pertos. Ea 0d ap -  anno de Uetiaa. 009,  1 Lam gue ct y Bertran "me terintoere y ly eno W entos hos ams aos ir Bar 00 da DRA Lenpts caretas y a. Jaelración. En el vegineo de prat, ben de RA, Desire es ql E yA de ade     Earle pe cier Pol mr merado, ore recado de Mess Bo que dira e dret A ja ue trends de ALO y quee Eeprom ra 20 a ee daa, meat sto 42 eo garito ya arpa de e pS Alerta, mbr ela dad a aa a pu ¿mece perio q. 1 aa Dis ro o, ag de nda de te on de nda era el jota        Guerra se precño Cuenca donde tempe qacida de bas vector, Era Deere ns ds porfis seta LE ps Eacseaa, Nocdaracca, eta, TES A eins cea to rpese a Daria amas electa pe        e rep Babris gore reparación, de qusrencrsa, prernes y unes     puede ingrid lares, Lada dem von y vet ie os pub pepa sore,     SA CTIERA QUE VIDA UTE LLLCUE HASTA EL 1071  Newmont desarrollará depósito de óxidos de oro Quecher Main  ss ac a dp Misa Pica rc ea 1 E ed nr ES E «e Enatca de rr eds Misco, bodies ha por de Tama Carat Treetuta eri ta0 dm o torbiscto paro le mejcca praia Be ds espai de     de Quiros Ml sea e el rado secre (Ceitaón A SS queria en dr el necios RAS  Ve val baza de Vaciar a ds "gros pe oe jet amd pres Dee repr mare,  de pera qe lev O  56 don barda de Queres Ile Mass ve eptrata aDOL, € 229 proderaiós Le Cispacacho     O Dry a se ratos,  Msrarietca da, dona de Mera es Tata, «cesan que el Sepa  Aur, Mire rg sroido ajenos entr? 5 eel rad Byrd, en Co ma a 7 gr haria pu     me de lea yc lego ee Var pitos, 1] Lego y ee See pos par Y ri de peas deca en el DA TEL de ha lererrracta da, Uagurera rtard serio need a reservas. Arieie e rgler aio, Capac pepe creta elo ás los ascii e Ya game ls Ln, gua e        de RO prat de oro po, jaca,  Pero a madero que Vamprocto Sil h ot er progecrd a marras! Lera dee Mess, mr icon a que precegía Dear a A AS  Laenpresa 2decea poo Lars AA 05 04 e pr eat, """,
    """Venta de vehículos espera repuntar en junio apoyada por la reconstrucción
    La venta de vehículos en el mercado de automóviles en el país ha experimentado un repunte en los últimos meses, según datos de la Asociación Nacional de Fabricantes de Automóviles (ANFAVEA). En el mes de junio, se registraron ventas de 12,654 unidades, un aumento del 10% con respecto al mes de mayo.
    La marca más vendida fue la Ford, con 2,500 unidades, seguida de la Toyota, con 2,000 unidades. Otros marcas que también registraron buenos resultados fueron la Chevrolet, con 1,500 unidades, y la Volkswagen, con 1,000 unidades.
    En cuanto a los tipos de vehículos, los automóviles fueron los más vendidos, con 9,000 unidades, seguidos de las camionetas con 3,000 unidades.
    La Asociación Nacional de Fabricantes de Automóviles (ANFAVEA) espera que la venta de vehículos continúe en alza en los próximos meses, gracias a la reconstrucción de la economía y la recuperación de la demanda.
    Newmont desarrollará depósito de óxidos de oro Quecher Main
    La empresa Newmont anunció que está desarrollando un depósito de óxidos de oro en el proyecto Quecher Main, ubicado en el distrito de Yacu Pampa, en la provincia de Cusco. El proyecto tiene una capacidad de producción de 100,000 toneladas de oro por año.
    El depósito de óxidos de oro Quecher Main es uno de los proyectos más importantes de Newmont en el Perú. La empresa espera que el proyecto genere empleo para miles de personas y contribuya al desarrollo económico de la región.
    La empresa ha invertido más de 100 millones de dólares en el proyecto, y espera que el primer ministro de producción de oro sea alcanzado en el año 2023.""",
    """PELIGROS DE LA EVA Y E
    LAS TOP TEN CON MEJOR GOBIERNO CORPORATIVO tengan
    20% de los que contribuyeron en los últimos 20 años de aporte y por
    activa, considera que el 10% de los que contribuyeron en los últimos 20 años de aporte y por Cae el precio mayorista del pollo, pero no a igual ritmo en mercados
    En los últimos siete días del mes, el margen comercial entre el precio mayorista y mayorista fue de 10%, según el INDEC. EL LUNES HAY PARO CONTRA PROYECTO
    Población participará en monitoreo ambiental de Tía María
    Pueblos podrán participar el próximo lunes para discutir y controlar el desarrollo de la mina que hará San Martín
    Se consideran plazas de trabajo según las personas que ganen el proyecto. Es el Ministerio de Energía y Minas, a través de la TIÓN
    CIENTE PLAZO DE APORTE DE 20 A 11 AÑOS PARA
    ¡SOBRE! FMI advierte que 60% de ONP no llegará
    Informe proyecta que cerca del 50% del Sistema Nacional de Pensiones no alcanzará el 100% de los beneficios previstos en las reformas aprobadas en el Congreso. EL diario de economía y negocios del Perú
    SOLANO ABELLANO
    "A la gran población no le preocupan las medidas reformistas" terie
    o de afili El sistema de pensiones en Alemania
    Los empleados a tiempo completo que han trabajado durante los últimos cinco años al Sistema de Pensiones contributivo y no contributivo se quedaban sin pensión.
    14 de marzo de 2018
    LOS DATOS
    Las celebridades que más ganan
    | Nombre         | Edad | Puesto       | Ganancias |
    |----------------|------|--------------|-----------|
    | Taylor Swift   | 29   | Cantante     | US$ 150   |
    | Elton John     | 73   | Músico       | US$ 120   |
    | Enrique Iglesias| 62  | Cantante     | US$ 110   |
    | Justin Bieber  | 24   | Cantante     | US$ 100   |
    | Ed Sheeran     | 26   | Cantante     | US$ 90    |
    Algunos son altos en sueldo
    Subirá en 50% stock de metros cuadrados de oficinas coworking
    ¿BAMBA TASA DE INTERÉS?
    BCR ve probable que demanda interna crezca menos de lo previsto Los niveles de presión son altos para
    las autoridades y que las
    autoridades
    metropolitana 'los cuellos blancos', sino si su quiosco vende"
    Espero en marketing que las cosas marchen bien y la gran población sea una gran audiencia que nos se mida. Hay la idea que todavía para mañana pueda recorrer el mundo por el 19
    RICARDO BARRIOS JIMÉNEZ
    Ya ofrecen viviendas de 36 metros cuadrados"""
    ]

test_images = [
    "./data/images/test.jpg",
    "./data/images/test2.jpg",
    "./data/images/test3.jpg"
]

def main():
    model, client = get_client_model()

    start_total = time.time()
    # with ThreadPoolExecutor(max_workers=64) as executor:
    #     results = list(executor.map(lambda q: ask_llm(client, model, prompt, q), questions))
    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(executor.map(lambda q: ask_vlm(client, model, prompt_image, encode_image(q)), test_images))
    end_total = time.time()

    average_time = (end_total-start_total) / len(results)
    print(f"Total time for all questions: {end_total - start_total:.2f}s")
    print(f"\nAverage time per question: {average_time:.2f}s")

    # for i, result in enumerate(results):
    #     print(f"\nResult for question {i+1}:\n{result.strip()}\n")

if __name__ == "__main__":
    main()

# numactl --cpunodebind=0 --membind=0 python3 -m vllm.entrypoints.openai.api_server --model data/quantizied_models/phi-4-Q4_K_M.gguf --tokenizer microsoft/Phi-4 --quantization gguf --tensor-parallel-size 1 --max-num-seqs 2048 --gpu-memory-utilization 0.98 --enable-prefix-caching --enforce-eager --swap-space 0 --max-model-len 5000 --max-num-batched-tokens 8192 --block-size 128 --disable-log-requests --disable-log-stats --disable-sliding-window --api-server-count 8
# numactl --cpunodebind=0 --membind=0 python3 -m vllm.entrypoints.openai.api_server --model nanonets/Nanonets-OCR-s --tensor-parallel-size 1 --max-num-seqs 4096 --gpu-memory-utilization 0.97 --enable-prefix-caching --enforce-eager --swap-space 0 --max-model-len 5000 --max-num-batched-tokens 16384 --block-size 128 --disable-log-requests --disable-log-stats
# numactl --cpunodebind=0 --membind=0 python3 -m vllm.entrypoints.openai.api_server --model reducto/RolmOCR --tensor-parallel-size 1 --max-num-seqs 4096 --gpu-memory-utilization 0.97 --enable-prefix-caching --enforce-eager --swap-space 0 --max-model-len 5000 --max-num-batched-tokens 16384 --block-size 128 --disable-log-requests --disable-log-stats
# numactl --cpunodebind=0 --membind=0 python3 -m vllm.entrypoints.openai.api_server --model allenai/olmOCR-7B-0725-FP8 --tensor-parallel-size 1 --max-num-seqs 4096 --gpu-memory-utilization 0.97 --enable-prefix-caching --enforce-eager --swap-space 0 --max-model-len 5000 --max-num-batched-tokens 16384 --block-size 128 --disable-log-requests --disable-log-stats
