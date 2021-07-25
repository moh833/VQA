import 'package:flutter/material.dart';

class QuestCards extends StatefulWidget {
  static const routeName = '/QuestCards';

  @override
  _QuestCardsState createState() => _QuestCardsState();
}

class _QuestCardsState extends State<QuestCards> {
  final _answerController = TextEditingController();
  var iscorrect = false;
  @override
  Widget build(BuildContext context) {
    Map<String, dynamic> card = ModalRoute.of(context).settings.arguments;
    //Navigator.of(context).pop('xyz');

    return MaterialApp(
      home: Scaffold(
        backgroundColor: Colors.black54,
        body: ListView(
          children: <Widget>[
            Stack(
              children: <Widget>[
                Container(
                  height: 800.0,
                  width: double.infinity,
                ),
                Container(
                  padding: EdgeInsets.all(10.0),
                  height: 500.0,
                  width: double.infinity,
                  decoration: BoxDecoration(
                      borderRadius: BorderRadius.only(
                        bottomLeft: Radius.circular(20.0),
                        bottomRight: Radius.circular(20.0),
                      ),
                      image: DecorationImage(
                        image: MemoryImage(card['image']),
                        fit: BoxFit.fill,
                      )),
                ),
                Positioned(
                  top: 420.0,
                  left: 10.0,
                  right: 10.0,
                  child: Material(
                    elevation: 10.0,
                    borderRadius: BorderRadius.circular(20.0),
                    child: Container(
                      height: 380.0,
                      decoration: BoxDecoration(
                          //borderRadius: BorderRadius.circular(20.0)
                          ),
                      padding: EdgeInsets.only(
                        left: 20.0,
                        right: 10.0,
                        top: 20.0,
                      ),
                      child: Column(
                        children: [
                          Container(
                            width: double.infinity,
                            child: Card(
                              elevation: 10,
                              shadowColor: Colors.purpleAccent,
                              child: Container(
                                padding: EdgeInsets.all(10.0),
                                width: double.infinity,
                                child: Column(
                                  children: [
                                    Container(
                                      width: double.infinity,
                                      child: Text(
                                        'Question:',
                                        textAlign: TextAlign.start,
                                        style: TextStyle(
                                            fontSize: 20.0,
                                            fontStyle: FontStyle.italic),
                                      ),
                                    ),
                                    Container(
                                      alignment: Alignment.topLeft,
                                      width: double.infinity,
                                      child: Text(
                                        card['question'].replaceAll("\n", ""),
                                        textAlign: TextAlign.left,
                                        style: TextStyle(
                                            fontSize: 20.0,
                                            fontStyle: FontStyle.italic),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                          Container(
                            padding: EdgeInsets.only(top: 10.0),
                            width: double.infinity,
                            child: Card(
                              elevation: 10,
                              shadowColor: Colors.purpleAccent,
                              child: Container(
                                padding: EdgeInsets.all(10.0),
                                width: double.infinity,
                                child: Column(
                                  children: [
                                    Container(
                                      width: double.infinity,
                                      child: Text(
                                        'Answer:',
                                        textAlign: TextAlign.start,
                                        style: TextStyle(
                                            fontSize: 20.0,
                                            fontStyle: FontStyle.italic),
                                      ),
                                    ),
                                    Container(
                                      alignment: Alignment.topLeft,
                                      width: double.infinity,
                                      child: Text(
                                        card['topAnswer'].replaceAll("\n", ""),
                                        textAlign: TextAlign.left,
                                        style: TextStyle(
                                            fontSize: 20.0,
                                            fontStyle: FontStyle.italic),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                          iscorrect
                              ? TextField(
                                  decoration:
                                      InputDecoration(labelText: 'Answer'),
                                  controller: _answerController,
                                  onChanged: (val) {
                                    card['topAnswer'] = val;
                                  },

                                  onSubmitted: (_) {
                                    setState(() {
                                      print(_answerController.text);
                                      card['topAnswer'] =
                                          _answerController.text;
                                    });
                                  },
                                  // onChanged: (val) => amountInput = val,
                                )
                              : Container(),
                          Row(
                            children: [
                              Container(
                                alignment: Alignment.centerRight,
                                child: OutlinedButton(
                                  style: OutlinedButton.styleFrom(
                                    shape: StadiumBorder(),
                                    side: BorderSide(
                                        width: 0, color: Colors.white),
                                  ),
                                  onPressed: () {
                                    setState(() {
                                      iscorrect = true;
                                    });
                                  },
                                  child: Icon(Icons.edit),
                                ),
                              ),
                              Spacer(),
                              OutlinedButton(
                                style: OutlinedButton.styleFrom(
                                  shape: StadiumBorder(),
                                  side:
                                      BorderSide(width: 0, color: Colors.white),
                                ),
                                onPressed: () {
                                  if (iscorrect == true) {
                                    print('pop from true');
                                    card['topAnswer'] = _answerController.text;
                                    Navigator.of(context)
                                        .pop(card['topAnswer']);
                                  } else {
                                    print('pop from false');
                                    Navigator.of(context)
                                        .pop(card['topAnswer']);
                                  }
                                },
                                child: Icon(Icons.done),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                )
              ],
            ),
          ],
        ),
      ),
    );
  }
}
