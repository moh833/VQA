import 'dart:convert';
import './quest.dart';
import './card.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() async {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.purple,
      ),

      home: MyHomePage(),
      initialRoute: '/', // default is '/'
      routes: {
        QuestCards.routeName: (ctx) => QuestCards(),
      },
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  var dataDone = true;
  var iscorrect = 'false';
  final passwordController = TextEditingController();
  Map<String, dynamic> out;
  List<Question> data = [];
  List<Widget> cardList = [];

  @override
  void initState() {
    getData().then((out) {
      setState(() {
        data = out;
        dataDone = false;
      });
    });

    // TODO: implement initState
    super.initState();
  }

  @override
  void didChangeDependencies() {
    // TODO: implement didChangeDependencies
    super.didChangeDependencies();
  }

  Future getData() async {
    final urs = 'https://shoofli-315622-default-rtdb.firebaseio.com/data.json';
    Uri erl = Uri.parse(urs) as Uri;
    var dataOut = await http.get(erl);
    var extracted = await jsonDecode(dataOut.body) as Map<String, dynamic>;
    List<Question> data1 = [];
    extracted.forEach((questId, questdata) {
      if (questdata['correct'] == '2') {
        print(questId);
        data1.add(Question(
          id: questId,
          image: Base64Decoder().convert(questdata['image']),
          correct: questdata['correct'],
          topAnswer: questdata['topAnswer'],
          question: questdata['question'],
        ));
      }
    });
    print(data1[0].image);

    return data1;
  }

  List<Widget> getdata(List<Question> data) {
    List<Widget> cardList1 = [];
    var lenth = data.length;
    print(lenth);
    if (lenth >= 5) {
      for (int i = 0; i < 5; i++) {
        double x = (5 - i) * 10.15;
        cardList1.add(Positioned(
          top: x,
          child: Draggable(
              onDraggableCanceled: (velocity, offset) {
                print(offset);
                if (offset > Offset(10, 5)) {
                  removeCards(data[i].id, data[i].topAnswer, i);
                }
              },
              childWhenDragging: Container(),
              feedback: GestureDetector(
                onTap: () {},
                child: Card(
                  elevation: 8.0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20.0),
                  ),
                  // color: Color.fromARGB(250, 112, 19, 179),
                  child: Column(
                    children: <Widget>[
                      Hero(
                        tag: "imageTag",
                        child: Image.memory(
                          data[i].image,
                          width: 320.0,
                          height: 440.0,
                          fit: BoxFit.fill,
                        ),
                      ),
                      Container(
                        padding: EdgeInsets.only(top: 10.0, bottom: 10.0),
                        child: Text(
                          data[i].topAnswer,
                          style: TextStyle(
                            fontSize: 20.0,
                            color: Colors.purple,
                          ),
                        ),
                      )
                    ],
                  ),
                ),
              ),
              child: GestureDetector(
                onTap: () {
                  Map<String, dynamic> gotox = {
                    'image': data[i].image,
                    'correct': data[i].correct,
                    'topAnswer': data[i].topAnswer,
                    "question": data[i].question,
                    "id": data[i].id
                  };
                  Navigator.of(context)
                      .pushNamed(
                    QuestCards.routeName,
                    arguments: gotox,
                  )
                      .then((value) {
                    if (value is String) {
                      print(value);
                      setState(() {
                        data[i].topAnswer = value;
                      });
                    }
                  });
                },
                child: Card(
                    elevation: 8.0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20.0),
                    ),
                    // color: Color.fromARGB(250, 112, 19, 179),
                    child: Column(
                      children: <Widget>[
                        Container(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.only(
                                topLeft: Radius.circular(20.0),
                                topRight: Radius.circular(20.0)),
                            image: DecorationImage(
                                image: MemoryImage(data[i].image),
                                fit: BoxFit.cover),
                          ),
                          height: 480.0,
                          width: 320.0,
                        ),
                        Container(
                          padding: EdgeInsets.only(top: 1.0, bottom: 10.0),
                          child: Text(
                            data[i].question.replaceAll("\n", ""),
                            style: TextStyle(
                              fontSize: 20.0,
                              color: Colors.purple,
                            ),
                          ),
                        )
                      ],
                    )),
              )),
        ));
      }
    } else if (lenth < 5) {
      for (int i = 0; i < data.length; i++) {
        double x = (data.length - i) * 10.15;
        cardList1.add(Positioned(
          top: x,
          child: Draggable(
              onDragEnd: (drag) {
                removeCards(data[i].id, data[i].topAnswer, i);
              },
              childWhenDragging: Container(),
              feedback: GestureDetector(
                child: Card(
                  elevation: 8.0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20.0),
                  ),
                  // color: Color.fromARGB(250, 112, 19, 179),
                  child: Column(
                    children: <Widget>[
                      Hero(
                        tag: "imageTag",
                        child: Image.memory(
                          data[i].image,
                          width: 320.0,
                          height: 440.0,
                          fit: BoxFit.fill,
                        ),
                      ),
                      Container(
                        padding: EdgeInsets.only(top: 10.0, bottom: 10.0),
                        child: Text(
                          data[i].topAnswer,
                          style: TextStyle(
                            fontSize: 20.0,
                            color: Colors.purple,
                          ),
                        ),
                      )
                    ],
                  ),
                ),
              ),
              child: GestureDetector(
                onTap: () {
                  Map<String, dynamic> gotox = {
                    'image': data[i].image,
                    'correct': data[i].correct,
                    'topAnswer': data[i].topAnswer,
                    "question": data[i].question,
                    "id": data[i].id
                  };
                  Navigator.of(context)
                      .pushNamed(
                    QuestCards.routeName,
                    arguments: gotox,
                  )
                      .then((value) {
                    if (value is String) {
                      print(value);
                      setState(() {
                        data[i].topAnswer = value;
                      });
                    }
                  });
                },
                child: Card(
                    elevation: 8.0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20.0),
                    ),
                    // color: Color.fromARGB(250, 112, 19, 179),
                    child: Column(
                      children: <Widget>[
                        Container(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.only(
                                topLeft: Radius.circular(20.0),
                                topRight: Radius.circular(20.0)),
                            image: DecorationImage(
                                image: MemoryImage(data[i].image),
                                fit: BoxFit.cover),
                          ),
                          height: 480.0,
                          width: 320.0,
                        ),
                        Container(
                          padding: EdgeInsets.only(top: 1.0, bottom: 10.0),
                          child: Column(
                            children: [
                              Text(
                                data[i].question.replaceAll("\n", "") + '?',
                                style: TextStyle(
                                  fontSize: 20.0,
                                  color: Colors.purple,
                                ),
                              ),
                            ],
                          ),
                        )
                      ],
                    )),
              )),
        ));
      }
    }

    List<Widget> cardlist = List.from(cardList1.reversed);
    return cardlist;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.black,
        ),
        drawer: Drawer(
          child: ListView(
            padding: EdgeInsets.zero,
            children: <Widget>[
              DrawerHeader(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: <Widget>[
                    Image.asset(
                      'assets/logo.png',
                      width: 80,
                      height: 80,
                    ),
                    SizedBox(
                      height: 15,
                    ),
                    Text(
                      "Shoofli Admin",
                      style: TextStyle(color: Colors.grey),
                    )
                  ],
                ),
                decoration: BoxDecoration(
                  color: Colors.white,
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: TextField(
                    obscureText: true,
                    decoration: InputDecoration(labelText: 'Password'),
                    controller: passwordController),
              ),
              ListTile(
                leading: Icon(Icons.pages),
                title: Text('Generate CSV'),
                onTap: () {
                  genCSV(passwordController.text).then((value) {
                    print(value);
                    if (value == 'done') {
                      Text('done');
                    } else {
                      Text('Password is not correct');
                    }
                  });
                  //Navigator.pop(context);
                },
              ),
              Center(
                  child: iscorrect == '1'
                      ? Text('Done')
                      : iscorrect == '2'
                          ? Text('Password is not correct')
                          : Container())
            ],
          ),
        ),
        backgroundColor: Colors.black,
        body: Center(
            child: dataDone
                ? CircularProgressIndicator()
                : Padding(
                    padding: const EdgeInsets.all(15),
                    child: Stack(
                        alignment: Alignment.center, children: getdata(data)),
                  )));
  }

  Future genCSV(String pass) async {
    Future.delayed(Duration(seconds: 0), () async {
      final urs = 'http://20.85.240.203:5000/auth';
      final response = await http.post(
        Uri.parse(urs),
        body: jsonEncode(
          {'pass': pass},
        ),
        headers: {'Content-Type': "application/json"},
      );
      print(response.body);
      if (response.body == 'done') {
        setState(() {
          iscorrect = '1';
        });
      } else {
        setState(() {
          iscorrect = '2';
        });
      }

      return response.body;
    });
  }

  Future removeCards(String i, dataget, index) async {
    Future.delayed(Duration(seconds: 0), () async {
      final urs = 'http://20.85.240.203:5000/update';
      final response = await http.post(
        Uri.parse(urs),
        body: jsonEncode(
          {'id': i, 'topAnswer': dataget},
        ),
        headers: {'Content-Type': "application/json"},
      ).then((value) {
        setState(() {
          data.removeAt(index);
        });
      });
    });
  }
}
