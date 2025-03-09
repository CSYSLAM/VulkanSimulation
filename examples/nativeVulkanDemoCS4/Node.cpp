#include "Node.h"
#include "Action.h"

#include "SceneGraph.h"

namespace CsyVk
{
Node::Node()
	: OBase()
	, m_node_name("default")
	, mDt(0.016f)
{
}


Node::~Node()
{
	mModuleList.clear();

	for (auto port : mExportNodes)
	{
		this->disconnect(port);
	}

	mImportNodes.clear();
	mExportNodes.clear();
}

void Node::setName(std::string name)
{
	m_node_name = name;
}

std::string Node::getName()
{
	return m_node_name;
}

std::string Node::getNodeType()
{
	return "Default";
}

bool Node::isAutoSync()
{
	return mAutoSync;
}

bool Node::isAutoHidden()
{
	return mAutoHidden;
}

void Node::setAutoSync(bool con)
{
	mAutoSync = con;
}

void Node::setAutoHidden(bool con)
{
	mAutoHidden = con;
}

bool Node::isActive()
{
	return mPhysicsEnabled;
}

void Node::setActive(bool active)
{
	mPhysicsEnabled = active;
}

bool Node::isVisible()
{
	return mRenderingEnabled;
}

void Node::setVisible(bool visible)
{
	mRenderingEnabled = visible;
}

float Node::getDt()
{
	return mDt;
}

void Node::setDt(Real dt)
{
	mDt = dt;
}

void Node::setSceneGraph(SceneGraph* scn)
{
	mSceneGraph = scn;
}

SceneGraph* Node::getSceneGraph()
{
	return mSceneGraph;
}

void Node::preUpdateStates()
{

}

void Node::updateStates()
{
	this->animationPipeline()->update();
}

void Node::update()
{
	if (!this->validateInputs()) {
		return;
	}

	if (this->requireUpdate())
	{
		this->preUpdateStates();

		if (mPhysicsEnabled) {
			this->updateStates();
		}

		this->postUpdateStates();

		this->updateTopology();

		//reset parameters
		for (auto param : fields_param)
		{
			param->tack();
		}

		//reset input fields
		for (auto f_in : fields_input)
		{
			f_in->tack();
		}

		//tag all output fields as modifed
		for (auto f_out : fields_output)
		{
			f_out->tick();
		}
	}
}

void Node::reset()
{
	if (this->validateInputs()) {
		this->stateElapsedTime()->setValue(0.0f);
		this->stateFrameNumber()->setValue(0);

		this->resetStates();

		//When the node is reset, call tick() to force updating all modules
		this->tick();
	}
}

NBoundingBox Node::boundingBox()
{
	return NBoundingBox();
}

void Node::postUpdateStates()
{

}

void Node::updateGraphicsContext()
{
	if (mRenderingEnabled)
	{
		this->graphicsPipeline()->update();
	}
}

void Node::resetStates()
{
	this->resetPipeline()->update();
}

bool Node::validateInputs()
{
	//If any input field is empty, return false;
	for(auto f_in : fields_input)
	{
		if (!f_in->isOptional() && f_in->isEmpty())
		{
			std::string errMsg = std::string("The field ") + f_in->getObjectName() +
				std::string(" in Node ") + this->getClassInfo()->getClassName() + std::string(" is not set!");

			Log::sendMessage(Log::Info, errMsg);
			return false;
		}
	}

	return true;
}

bool Node::requireUpdate()
{
	//TODO: improve the following rules
	if (mForceUpdate)
		return true;

	//check input fields
	bool modified = false;

	if (mImportNodes.size() > 0)
	{
		return true;
	}
	 

	for (auto f_in : fields_input)
	{
		modified |= f_in->isModified();
	}

	//check control fields
	for (auto var : fields_param)
	{
		modified |= var->isModified();
	}

	return modified;
}

void Node::tick()
{
	std::vector<FBase*>& fields = this->getAllFields();
	for(FBase * var : fields)
	{
		if (var != nullptr) {
			if (var->getFieldType() == FieldTypeEnum::State || var->getFieldType() == FieldTypeEnum::Out)
			{
				var->tick();
			}
		}
	}
}

std::shared_ptr<Pipeline> Node::resetPipeline()
{
	if (mResetPipeline == nullptr)
	{
		mResetPipeline = std::make_shared<AnimationPipeline>(this);
	}
	return mResetPipeline;
}

std::shared_ptr<AnimationPipeline> Node::animationPipeline()
{
	if (mAnimationPipeline == nullptr)
	{
		mAnimationPipeline = std::make_shared<AnimationPipeline>(this);
	}
	return mAnimationPipeline;
}

std::shared_ptr<GraphicsPipeline> Node::graphicsPipeline()
{
	if (mGraphicsPipeline == nullptr)
	{
		mGraphicsPipeline = std::make_shared<GraphicsPipeline>(this);
	}
	return mGraphicsPipeline;
}

bool Node::addModule(std::shared_ptr<Module> module)
{
	bool ret = true;
	ret &= addToModuleList(module);

	return ret;
}

bool Node::deleteModule(std::shared_ptr<Module> module)
{
	bool ret = true;

	ret &= deleteFromModuleList(module);
		
	return ret;
}

std::string FormatConnectionInfo(Node* node, NodePort* port, bool connecting, bool succeeded)
{
	Node* pOut = port != nullptr ? port->getParent() : nullptr;

	std::string capIn = node->caption();
	std::string capOut = pOut != nullptr ? pOut->caption() : "";

	std::string nameIn = node->getName();
	std::string nameOut = port != nullptr ? port->getPortName() : "";

	if (connecting)
	{
		std::string message1 = capIn + ":" + nameIn + " is connected to " + capOut + ":" + nameOut;
		std::string message2 = capIn + ":" + nameIn + " cannot be connected to " + capOut + ":" + nameOut;
		return succeeded ? message1 : message2;
	}
	else
	{
		std::string message1 = capIn + ":" + nameIn + " is disconnected from " + capOut + ":" + nameOut;
		std::string message2 = capIn + ":" + nameIn + " cannot be disconnected from " + capOut + ":" + nameOut;
		return succeeded ? message1 : message2;
	}
}

bool Node::appendExportNode(NodePort* nodePort)
{
	auto it = find(mExportNodes.begin(), mExportNodes.end(), nodePort);
	if (it != mExportNodes.end()) {
		Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, true, false));
		return false;
	}

	mExportNodes.push_back(nodePort);

	//Always show the last node
	if (mAutoHidden)
		this->setVisible(false);

	Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, true, true));
	return nodePort->addNode(this);
}

bool Node::removeExportNode(NodePort* nodePort)
{
	//TODO: this is a hack, otherwise the app will crash
	if (mExportNodes.size() == 0) {
		return false;
	}

	auto it = find(mExportNodes.begin(), mExportNodes.end(), nodePort);
	if (it == mExportNodes.end()) {
		Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, false, false));
		return false;
	}

	mExportNodes.erase(it);

	//Recover the visibility
	if (mAutoHidden)
		this->setVisible(true);

	Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, false, true));
	return nodePort->removeNode(this);
}

void Node::updateTopology()
{

}

bool Node::connect(NodePort* nPort)
{
	nPort->notify();

	return this->appendExportNode(nPort);
}

bool Node::disconnect(NodePort* nPort)
{
	return this->removeExportNode(nPort);
}

bool Node::attachField(FBase* field, std::string name, std::string desc, bool autoDestroy /*= true*/)
{
	field->setParent(this);
	field->setObjectName(name);
	field->setDescription(desc);
	field->setAutoDestroy(autoDestroy);

	bool ret = false;
	
	auto fType = field->getFieldType();
	switch (field->getFieldType())
	{
	case FieldTypeEnum::State:
		ret = this->addField(field);
		break;

	case FieldTypeEnum::Param:
		ret = addParameter(field);
		break;

	case FieldTypeEnum::In:
		ret = addInputField(field);
		break;

	case FieldTypeEnum::Out:
		ret = addOutputField(field);
		break;

	default:
		break;
	}
	

	if (!ret)
	{
		Log::sendMessage(Log::Error, std::string("The field ") + name + std::string(" already exists!"));
	}
	return ret;
}

uint Node::sizeOfImportNodes() const
{
	uint n = 0;
	for(auto port : mImportNodes)
	{
		n += port->getNodes().size();
	}

	return n;
}

void Node::setForceUpdate(bool b)
{
	mForceUpdate = b;
}

// Node* Node::addDescendant(Node* descent)
// {
// 	if (hasDescendant(descent) || descent == nullptr)
// 		return descent;
// 
// 	mDescendants.push_back(descent);
// 	return descent;
// }
// 
// bool Node::hasDescendant(Node* descent)
// {
// 	auto it = std::find(mDescendants.begin(), mDescendants.end(), descent);
// 	return it == mDescendants.end() ? false : true;
// }
// 
// void Node::removeDescendant(Node* descent)
// {
// 	auto iter = mDescendants.begin();
// 	for (; iter != mDescendants.end(); )
// 	{
// 		if (*iter == descent)
// 		{
// 			mDescendants.erase(iter++);
// 		}
// 		else
// 		{
// 			++iter;
// 		}
// 	}
// }

bool Node::addNodePort(NodePort* port)
{
	mImportNodes.push_back(port);

	return true;
}

// void Node::setAsCurrentContext()
// {
// 	getContext()->enable();
// }

// void Node::setTopologyModule(std::shared_ptr<TopologyModule> topology)
// {
// 	if (m_topology != nullptr)
// 	{
// 		deleteModule(m_topology);
// 	}
// 	m_topology = topology;
// 	addModule(topology);
// }
// 
// void Node::setNumericalModel(std::shared_ptr<NumericalModel> numerical)
// {
// 	if (m_numerical_model != nullptr)
// 	{
// 		deleteModule(m_numerical_model);
// 	}
// 	m_numerical_model = numerical;
// 	addModule(numerical);
// }
// 
// void Node::setCollidableObject(std::shared_ptr<CollidableObject> collidable)
// {
// 	if (m_collidable_object != nullptr)
// 	{
// 		deleteModule(m_collidable_object);
// 	}
// 	m_collidable_object = collidable;
// 	addModule(collidable);
// }

std::shared_ptr<Module> Node::getModule(std::string name)
{
	std::shared_ptr<Module> base = nullptr;
	std::list<std::shared_ptr<Module>>::iterator iter;
	for (iter = mModuleList.begin(); iter != mModuleList.end(); iter++)
	{
		if ((*iter)->getName() == name)
		{
			base = *iter;
			break;
		}
	}
	return base;
}

bool Node::hasModule(std::string name)
{
	if (getModule(name) == nullptr)
		return false;

	return true;
}

/*Module* Node::getModule(std::string name)
{
	std::map<std::string, Module*>::iterator result = m_modules.find(name);
	if (result == m_modules.end())
	{
		return NULL;
	}

	return result->second;
}*/


bool Node::addToModuleList(std::shared_ptr<Module> module)
{
	auto found = std::find(mModuleList.begin(), mModuleList.end(), module);
	if (found == mModuleList.end())
	{
		mModuleList.push_back(module);
		module->setParentNode(this);
		return true;
	}

	return false;
}

bool Node::deleteFromModuleList(std::shared_ptr<Module> module)
{
	auto found = std::find(mModuleList.begin(), mModuleList.end(), module);
	if (found != mModuleList.end())
	{
		mModuleList.erase(found);
		return true;
	}

	return true;
}

}