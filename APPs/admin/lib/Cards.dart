import 'package:flutter/material.dart';
import './card.dart';

class Cards extends StatefulWidget {
  static const routeName = '/cards';
  final dynamic image;
  final String correct;
  String topAnswer;
  final String question;
  final String id;

  double margin;
  Cards(
      {this.image,
      this.correct,
      this.topAnswer,
      this.question,
      this.id,
      this.margin});

  @override
  _CardsState createState() => _CardsState();
}

class _CardsState extends State<Cards> {
  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: widget.margin,
      child: Draggable(
          onDragEnd: (drag) {
            removeCards(widget.id);
          },
          childWhenDragging: Container(),
          feedback: GestureDetector(
            onTap: () {
              print("Hello All");
            },
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
                      widget.image,
                      width: 320.0,
                      height: 440.0,
                      fit: BoxFit.fill,
                    ),
                  ),
                  Container(
                    padding: EdgeInsets.only(top: 10.0, bottom: 10.0),
                    child: Text(
                      widget.topAnswer,
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
                'image': widget.image,
                'correct': widget.correct,
                'topAnswer': widget.topAnswer,
                "question": widget.question,
                "id": widget.id
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
                    widget.topAnswer = value;
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
                            image: MemoryImage(widget.image),
                            fit: BoxFit.cover),
                      ),
                      height: 480.0,
                      width: 320.0,
                    ),
                    Container(
                      padding: EdgeInsets.only(top: 1.0, bottom: 10.0),
                      child: Text(
                        widget.question.replaceAll("\n", ""),
                        style: TextStyle(
                          fontSize: 20.0,
                          color: Colors.purple,
                        ),
                      ),
                    )
                  ],
                )),
          )),
    );
  }

  Future removeCards(String id) {
    print("Hello All");
  }
}
