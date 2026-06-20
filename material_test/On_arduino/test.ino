int pin_l = 12;

void setup() {
  pinMode(pin_l, OUTPUT);
  Serial.begin(9600);
  Serial.println("Envoyez 't' pour lancer le test de performance.");
}

void loop() {
        
    digitalWrite(pin_l, HIGH);
    
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 't') {
      unsigned long start = micros();
      for(int i = 0; i < 1000; i++) {
        digitalWrite(pin_l, LOW);
        digitalWrite(pin_l, HIGH);
      }
      unsigned long end = micros();
      unsigned long duration = end - start;
      Serial.print("Temps pour 1000 bascules: ");
      Serial.print(duration);
      Serial.println(" microsecondes");
    } else {
      Serial.println("Commande inconnue. Envoyez 't' pour tester.");
    }
  }
   digitalWrite(pin_l, LOW);
}