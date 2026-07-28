import json
import urllib.request
import urllib.parse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "textbooks"

SAMPLES = [
    {
        "standard": "10",
        "subject": "science",
        "chapter": "chemical_reactions",
        "wiki_title": "Chemical reaction",
        "fallback_text": """
NCERT Class 10 Science - Chapter 1: Chemical Reactions and Equations

A chemical reaction is a process that leads to the chemical transformation of one set of chemical substances to another. Classically, chemical reactions encompass changes that only involve the positions of electrons in the forming and breaking of chemical bonds between atoms, with no change to the nuclei, and can often be described by a chemical equation.

Characteristics of Chemical Reactions:
1. Evolution of a gas.
2. Change in temperature.
3. Formation of a precipitate.
4. Change in color.
5. Change in state.

Chemical Equations:
A chemical equation is the symbolic representation of a chemical reaction in the form of symbols and formulae, wherein the reactant entities are given on the left-hand side and the product entities on the right-hand side.

Types of Chemical Reactions:
- Combination Reaction: Two or more reactants combine to form a single product. (e.g., burning of coal)
- Decomposition Reaction: A single reactant breaks down to give two or more simpler products.
- Displacement Reaction: A more reactive element displaces a less reactive element from its compound.
- Double Displacement Reaction: Two compounds exchange ions to form two new compounds.
- Oxidation and Reduction (Redox): Oxidation is the gain of oxygen or loss of hydrogen. Reduction is the loss of oxygen or gain of hydrogen.
        """
    },
    {
        "standard": "12",
        "subject": "physics",
        "chapter": "electrostatics",
        "wiki_title": "Electrostatics",
        "fallback_text": """
NCERT Class 12 Physics - Chapter 1 & 2: Electrostatics

Electrostatics is a branch of physics that studies electric charges at rest (static electricity). Since classical physics, it has been known that some materials, such as amber, attract lightweight particles after rubbing.

Electric Charge:
Charge is a fundamental property of matter. There are two types of charges: positive and negative. Like charges repel each other, and unlike charges attract. The SI unit of charge is the Coulomb (C). Charge is quantized, meaning any charge q is an integral multiple of the elementary charge e (q = ne, where e = 1.6 x 10^-19 C).

Coulomb's Law:
The force of attraction or repulsion between two stationary point charges is directly proportional to the product of the charges and inversely proportional to the square of the distance between them.
F = k * (q1 * q2) / r^2, where k is Coulomb's constant (9 x 10^9 N m^2 C^-2).

Electric Field:
The electric field is defined as the electric force per unit charge. The direction of the field is taken to be the direction of the force it would exert on a positive test charge.
E = F / q.

Electric Potential:
Electric potential is the amount of work energy needed to move a unit of electric charge from a reference point to a specific point in an electric field without producing an acceleration.
        """
    },
    {
        "standard": "08",
        "subject": "science",
        "chapter": "light",
        "wiki_title": "Light",
        "fallback_text": """
NCERT Class 8 Science - Chapter 16: Light

Light is electromagnetic radiation that can be detected by the human eye. The primary source of light on Earth is the Sun.

Reflection of Light:
When light falls on a polished or shiny surface, it bounces back into the same medium. This phenomenon is called reflection of light.

Laws of Reflection:
1. The angle of incidence is always equal to the angle of reflection. (Angle i = Angle r).
2. The incident ray, the normal at the point of incidence, and the reflected ray all lie in the same plane.

Types of Reflection:
- Regular Reflection: When light reflects off a smooth, polished surface (like a mirror), producing a clear image.
- Diffused or Irregular Reflection: When light reflects off a rough surface, dispersing in different directions, and does not form a clear image.

Human Eye:
The human eye is one of our most valuable and sensitive sense organs. It enables us to see the wonderful world and the colors around us.
Key parts of the eye:
- Cornea: Outer transparent protective layer.
- Iris: Colored part that controls pupil size.
- Pupil: Aperture through which light enters.
- Lens: Focuses light on the retina.
- Retina: Screen containing light-sensitive rods and cones.
        """
    }
]

def fetch_wiki_text(title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={urllib.parse.quote(title)}&format=json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (BlindAssist NCERT Downloader)'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if "extract" in page:
                    return page["extract"]
    except Exception as e:
        print(f"Network request failed for '{title}': {e}. Using fallback local text.")
    return None

def main():
    print("📥 Starting NCERT Sample Downloader...")
    for item in SAMPLES:
        std_dir = DATA_DIR / f"standard_{item['standard'].zfill(2)}" / item["subject"]
        std_dir.mkdir(parents=True, exist_ok=True)
        file_path = std_dir / f"{item['chapter']}.txt"
        
        print(f"Checking {file_path.relative_to(BASE_DIR)}...")
        
        # Try fetching from Wikipedia
        text = fetch_wiki_text(item["wiki_title"])
        if not text:
            text = item["fallback_text"]
        else:
            # Add header to the fetched wiki text
            text = f"NCERT Class {item['standard']} {item['subject'].capitalize()} - Chapter: {item['chapter'].replace('_', ' ').title()}\n\n" + text
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text.strip() + "\n")
            
        print(f"✅ Successfully wrote chapter to {file_path.name}")
        
    print("\n🎉 Download / creation of NCERT sample chapters complete!")

if __name__ == '__main__':
    main()
