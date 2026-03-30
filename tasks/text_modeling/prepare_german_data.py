import os

def create_german_conversational_dataset(data_dir):
    file_path = os.path.join(data_dir, "german_conversational.txt")
    
    # Ein Korpus an allgemeinen deutschen Sätzen für Unterhaltungen
    sentences = [
        "Hallo, wie geht es dir heute?",
        "Mir geht es gut, danke der Nachfrage.",
        "Was hast du heute so gemacht?",
        "Ich bin heute früh aufgestanden und habe Kaffee getrunken.",
        "Das Wetter ist heute wirklich schön, die Sonne scheint.",
        "Hast du Lust, später etwas essen zu gehen?",
        "Ich würde gerne ins Kino gehen, läuft etwas Gutes?",
        "Kannst du mir bitte helfen, ich verstehe das nicht.",
        "Es ist wichtig, dass wir miteinander reden.",
        "Ich freue mich sehr, dich kennenzulernen.",
        "Was sind deine Hobbys?",
        "Ich lese gerne Bücher und gehe spazieren.",
        "Musik hören entspannt mich nach einem langen Tag.",
        "Woher kommst du ursprünglich?",
        "Ich wohne schon seit vielen Jahren in dieser Stadt.",
        "Hast du Geschwister?",
        "Ja, ich habe einen Bruder und eine Schwester.",
        "Was arbeitest du von Beruf?",
        "Ich bin Student und lerne gerade viel für meine Prüfungen.",
        "Reisen ist eine meiner größten Leidenschaften.",
        "Ich war letztes Jahr in Italien im Urlaub.",
        "Kannst du mir den Weg zum Bahnhof erklären?",
        "Entschuldigung, wie spät ist es gerade?",
        "Ich glaube, es wird bald regnen, wir sollten reingehen.",
        "Hast du schon Pläne für das Wochenende?",
        "Ich werde mich einfach nur ausruhen und entspannen.",
        "Das Essen schmeckt wirklich hervorragend.",
        "Können wir uns morgen treffen?",
        "Ich rufe dich später noch einmal an.",
        "Vielen Dank für deine Hilfe, das war sehr nett.",
        "Ich wünsche dir einen schönen Tag!",
        "Gute Nacht und schlaf gut.",
        "Wie war dein Tag auf der Arbeit?",
        "Es war ziemlich stressig, aber ich habe viel geschafft.",
        "Was ist dein Lieblingsessen?",
        "Ich mag Pizza und Pasta sehr gerne.",
        "Treibst du regelmäßig Sport?",
        "Ja, ich gehe zweimal die Woche joggen.",
        "Das ist eine interessante Frage, darüber muss ich nachdenken.",
        "Ich stimme dir voll und ganz zu.",
        "Das sehe ich etwas anders, aber ich verstehe deinen Punkt.",
        "Lass uns das Thema wechseln.",
        "Hast du den neuen Film schon gesehen?",
        "Nein, aber ich habe gehört, er soll gut sein.",
        "Ich lerne gerade eine neue Sprache.",
        "Es macht Spaß, neue Dinge auszuprobieren.",
        "Familie und Freunde sind mir sehr wichtig.",
        "Manchmal brauche ich einfach etwas Zeit für mich allein.",
        "Das Leben ist voller Überraschungen.",
        "Man lernt nie aus."
    ]
    
    print(f"Erstelle deutschen Konversations-Datensatz in {file_path}...")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        # Wir wiederholen die Sätze oft, um genug Daten für ein kurzes Training zu haben
        for _ in range(500): 
            for sentence in sentences:
                f.write(sentence + "\n")
                
    print(f"Datei erstellt. Größe: {os.path.getsize(file_path) / 1024:.2f} KB")
    return file_path

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    create_german_conversational_dataset(data_dir)
