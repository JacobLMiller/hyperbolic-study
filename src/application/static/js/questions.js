
function appendQuestion(option, optionText){
    var parentChoices = document.createElement("div")
    parentChoices.className = "option-container";

    var answerNode = document.createElement("input");
    answerNode.setAttribute("type", "radio");
    answerNode.setAttribute('id', option);
    answerNode.setAttribute('name', "answers");

    var answerText = document.createElement("label");
    answerText.setAttribute('for', option);
    answerText.textContent = optionText;

    parentChoices.appendChild(answerNode);
    parentChoices.appendChild(answerText);
    return parentChoices;
}

function setupQuestionAndInteractions(vis, question_obj, questionID){

    vis.interact(question_obj["node"]);    
    vis.highlight_question(question_obj["node"]);

    let question = document.getElementById("question-form");
    if (questionID) {

        question.innerHTML = question_obj["q_text"];
        var question_options = Object.keys(question_obj["q_options"]);
        questionChoices = document.getElementById("question-options-form");

        for (var i = 0; i < question_options.length; i++) {
            let parentChoices = appendQuestion(question_options[i], question_obj["q_options"][question_options[i]]);

            questionChoices.appendChild(parentChoices);

            document.getElementById(question_options[i]).addEventListener("change", e => {
                question.value = e.target.id;
            })
        }

        var submitButton = document.createElement("input")
        submitButton.setAttribute("id", "submit-button");
        submitButton.setAttribute("type", "submit")
        submitButton.classList.add("submit-button")
        questionChoices.appendChild(submitButton);

    }
}