---
layout: default
title: Schedule
---

# Schedule
----------

<p>* = optional reading</p>

<table class="table table-striped"> 
  <tbody>
    <tr>
      <th>Date</th>
      <th>Topic</th>
      <th>Readings / Videos</th>
	  <th width="15%">Assignments</th>
    </tr>
	
    {% for lecture in site.data.schedule.lectures %}
    <tr>
        <td>{{ lecture.date | date: "%d %b" }}</td>
        <td>
		   {% for lect in lecture.content %}
		   <p class="{{ lecture.type }}">		  
	        {% if lect.slides %}
				<a href="{{ lect.slides }}">{{ lect.title }}</a>
            {% else %}
				{{ lect.title }}
			{% endif %}
		    </p>
		   {% endfor %}
        </td>
		
        <td>
          {% if lecture.reading %}
             <ul class="reading-list">
                {% for reading in lecture.reading %}
                   <li class="reading-entry">
                        {% if reading.optional %}
			   *
                        {% else %}
							<i class="fa-li fa"> </i>
						{% endif %}
                        {% if reading.author %}{{ reading.author }}, {% endif %}
                        {% if reading.url %}
						    <a href="{{ reading.url }}">{{ reading.title }}</a>
                        {% else %}
                            {{ reading.title }} 
                        {% endif %}
			{% if reading.note %} {{ reading.note }} {% endif %}
                   </li>
               {% endfor %}
          </ul>
        {% endif %}
      </td>

	  <td>
        {% for assignment in lecture.assignment %}
			{% if assignment.link %}
  			  <a href="{{ assignment.link }}">{{ assignment.label }}</a>
		    {% else %}
			  {{ assignment.label }}
	        {% endif %}
		{% endfor %}
	  </td>
    </tr>
    {% endfor %}
  </tbody>
</table>
