#define d_port    51
#define s_port    40
#define bit_rate  50
#define buf_size  3000
#define T_reset   800

double T_p = 1000 / bit_rate;

void setup() {
  String msg;
  // shut down the source
  pinMode(d_port, OUTPUT);
  digitalWrite(d_port, HIGH);
  // open the shutter
  pinMode(s_port, OUTPUT);
  digitalWrite(s_port, HIGH);

  Serial.begin(115200);
  while (!Serial) {;}
  Serial.println("Hello, Python");
  while(1){
    if (Serial.available() > 0){
      msg = Serial.readStringUntil('\n');
      if (msg == "Hello, Arduino")
      {break;}
    }
  }
}

void loop() {
  String msg;
  unsigned int len;
  byte frame_buf[buf_size];
  
  if (Serial.available() > 0) {
    msg = Serial.readStringUntil('\n');
    Serial.println("Ready");
    if (msg == "Data")
    {
        msg = Serial.readStringUntil('\n');
        len = msg.toInt();
        msg = Serial.readStringUntil('\n');
        msg.getBytes(frame_buf, len+1);
        send_frame(frame_buf, len);
    }
    if (msg == "Calibration")
    {
        calibrate();
    }
    if (msg == "br50")
    {
        T_p = 1000 / 50;
    }
    if (msg == "br80")
    {
        T_p = 1000 / 80;
    }
    if (msg == "br100")
    {
        T_p = 1000 / 100;
    }
    Serial.println("Done");
  }
}


void send_bit(byte x) {
  if (x == '1') {
    digitalWrite(d_port, LOW);
  }
  else {
    digitalWrite(d_port, HIGH);
  }
  delay(T_p);
  return;
}

void send_frame(byte frame[], unsigned int len) {
  for(int i = 0; i < len; i++){
    send_bit(frame[i]);
  }
  digitalWrite(d_port, HIGH);
  delay(T_reset);
  return;
}

void calibrate(void){
  // close the shutter
  digitalWrite(s_port, LOW);
  // wait until the shutter is fully closed
  delay(100);
  // open the source
  digitalWrite(d_port, LOW);
  // wait until the source reaches its steady state
  delay(1000);
  // open the shutter to generate a quasi step input
  digitalWrite(s_port, HIGH);
  // wait until the detector reaches its steady state and generate a complete step response
  delay(600);
  // close the source, wait until it recovers to room temperature
  digitalWrite(d_port, HIGH);
  delay(T_reset);
}
